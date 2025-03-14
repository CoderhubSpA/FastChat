"""
Usage:
python3 qa_browser.py --share
"""

import argparse
from collections import defaultdict
import re
import os
import gradio as gr

from fastchat.llm_judge.common import (
    load_questions,
    load_model_answers,
    load_single_model_judgments,
    load_pairwise_model_judgments,
    get_single_judge_explanation,
    get_pairwise_judge_explanation,
    NEED_REF_CATS
)


def resolve_pairwise_judgment_dict(
    question, model_judgments_normal, model_judgments_math, judge_model, multi_turn=False
):
    """Return the correct pairwise judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[(judge_model, "pair-math-v1-multi-turn")]
        return model_judgments_normal[(judge_model, "pair-v2-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[(judge_model, "pair-math-v1")]
    else:
        return model_judgments_normal[(judge_model, "pair-v2")]


def resolve_single_judgment_dict(
    question, model_judgments_normal, model_judgments_math, judge_model, multi_turn=False
):
    """Return the correct single answer grading judge."""
    if multi_turn:
        if question["category"] in NEED_REF_CATS:
            return model_judgments_math[(judge_model, "single-math-v1-multi-turn")]
        return model_judgments_normal[(judge_model, "single-v1-multi-turn")]

    if question["category"] in NEED_REF_CATS:
        return model_judgments_math[(judge_model, "single-math-v1")]
    else:
        return model_judgments_normal[(judge_model, "single-v1")]

questions = []
model_answers = {}

model_judgments_normal_single = {}
model_judgments_math_single = {}

model_judgments_normal_pairwise = {}
model_judgments_math_pairwise = {}

question_selector_map = {}
category_selector_map = defaultdict(list)


def display_question(category_selector, request: gr.Request):
    # Convert category_selector to string if it's a list
    if isinstance(category_selector, list) and len(category_selector) > 0:
        category_selector = category_selector[0]
    
    # Now category_selector should be a string
    choices = category_selector_map.get(category_selector, [])
    return gr.Dropdown(
        value=choices[0] if choices else None,
        choices=choices,
    )


def display_pairwise_answer(
    question_selector, model_selector1, model_selector2, judge_model, request: gr.Request
):
    q = question_selector_map[question_selector]
    qid = q["question_id"]

    ans1 = model_answers[model_selector1][qid]
    ans2 = model_answers[model_selector2][qid]

    chat_mds = pairwise_to_gradio_chat_mds(q, ans1, ans2)
    gamekey = (qid, model_selector1, model_selector2)

    # Only show first turn judgment if available
    explanation = ""
    if model_judgments_normal_pairwise:
        try:
            judgment_dict = resolve_pairwise_judgment_dict(
                q,
                model_judgments_normal_pairwise,
                model_judgments_math_pairwise,
                judge_model=judge_model,
                multi_turn=False,
            )
            explanation = (
                "##### Model Judgment (first turn)\n"
                + get_pairwise_judge_explanation(gamekey, judgment_dict)
            )
        except (KeyError, TypeError):
            explanation = "##### Model Judgment\nNo judgment available for the first turn."

    # Only show second turn judgment if multi-turn answers are available
    explanation_turn2 = ""
    if len(q.get("turns", [])) > 1 and model_judgments_normal_pairwise:
        try:
            judgment_dict_turn2 = resolve_pairwise_judgment_dict(
                q,
                model_judgments_normal_pairwise,
                model_judgments_math_pairwise,
                judge_model=judge_model,
                multi_turn=True,
            )
            explanation_turn2 = (
                "##### Model Judgment (second turn)\n"
                + get_pairwise_judge_explanation(gamekey, judgment_dict_turn2)
            )
        except (KeyError, TypeError):
            explanation_turn2 = "##### Model Judgment\nNo judgment available for the second turn."

    return chat_mds + [explanation] + [explanation_turn2]


def display_single_answer(
    question_selector, model_selector1, judge_model, request: gr.Request
):
    q = question_selector_map[question_selector]
    qid = q["question_id"]

    ans1 = model_answers[model_selector1][qid]

    chat_mds = single_to_gradio_chat_mds(q, ans1)
    gamekey = (qid, model_selector1)

    # Only show first turn judgment if available
    explanation = ""
    if model_judgments_normal_single:
        try:
            judgment_dict = resolve_single_judgment_dict(
                q, model_judgments_normal_single, model_judgments_math_single, 
                judge_model=judge_model, multi_turn=False
            )
            explanation = "##### Model Judgment (first turn)\n" + get_single_judge_explanation(
                gamekey, judgment_dict
            )
        except (KeyError, TypeError):
            explanation = "##### Model Judgment\nNo judgment available for the first turn."

    # Only show second turn judgment if multi-turn answers are available
    explanation_turn2 = ""
    if len(q.get("turns", [])) > 1 and model_judgments_normal_single:
        try:
            judgment_dict_turn2 = resolve_single_judgment_dict(
                q, model_judgments_normal_single, model_judgments_math_single, 
                judge_model=judge_model, multi_turn=True
            )
            explanation_turn2 = (
                "##### Model Judgment (second turn)\n"
                + get_single_judge_explanation(gamekey, judgment_dict_turn2)
            )
        except (KeyError, TypeError):
            explanation_turn2 = "##### Model Judgment\nNo judgment available for the second turn."

    return chat_mds + [explanation] + [explanation_turn2]


newline_pattern1 = re.compile(r"\n\n(\d+\. )")
newline_pattern2 = re.compile(r"\n\n(- )")


def post_process_answer(x):
    """Fix Markdown rendering problems."""
    x = x.replace("\u2022", "- ")
    x = re.sub(newline_pattern1, r"\n\g<1>", x)
    x = re.sub(newline_pattern2, r"\n\g<1>", x)
    return x


def pairwise_to_gradio_chat_mds(question, ans_a, ans_b, turn=None):
    # Handle single-turn vs multi-turn questions
    if not isinstance(question.get("turns", []), list):
        question["turns"] = [question.get("turns", "")]
    
    # Ensure turns exists in question
    if "turns" not in question:
        question["turns"] = [""]
    
    end = min(len(question["turns"]), 2)  # Display up to 2 turns
    if turn is not None:
        end = min(turn + 1, end)

    mds = ["", "", "", "", "", "", ""]
    
    for i in range(end):
        base = i * 3
        if i == 0:
            mds[base + 0] = "##### User\n" + question["turns"][i]
        else:
            mds[base + 0] = "##### User's follow-up question \n" + question["turns"][i]
        
        # Handle single-turn answers
        if "choices" in ans_a and "turns" in ans_a["choices"][0]:
            turns_a = ans_a["choices"][0]["turns"]
            if i < len(turns_a):
                mds[base + 1] = "##### Assistant A\n" + post_process_answer(
                    turns_a[i].strip()
                )
            else:
                mds[base + 1] = "##### Assistant A\n" + "No answer for this turn."
        else:
            # Single turn format
            if i == 0 and "choices" in ans_a and isinstance(ans_a["choices"], list) and ans_a["choices"]:
                if isinstance(ans_a["choices"][0], dict) and "turns" in ans_a["choices"][0]:
                    mds[base + 1] = "##### Assistant A\n" + post_process_answer(
                        ans_a["choices"][0]["turns"][0].strip() if ans_a["choices"][0]["turns"] else ""
                    )
                else:
                    mds[base + 1] = "##### Assistant A\n" + post_process_answer(
                        str(ans_a["choices"][0]).strip()
                    )
            else:
                mds[base + 1] = "##### Assistant A\n" + "No answer available."
        
        # Handle single-turn answers for model B
        if "choices" in ans_b and "turns" in ans_b["choices"][0]:
            turns_b = ans_b["choices"][0]["turns"]
            if i < len(turns_b):
                mds[base + 2] = "##### Assistant B\n" + post_process_answer(
                    turns_b[i].strip()
                )
            else:
                mds[base + 2] = "##### Assistant B\n" + "No answer for this turn."
        else:
            # Single turn format
            if i == 0 and "choices" in ans_b and isinstance(ans_b["choices"], list) and ans_b["choices"]:
                if isinstance(ans_b["choices"][0], dict) and "turns" in ans_b["choices"][0]:
                    mds[base + 2] = "##### Assistant B\n" + post_process_answer(
                        ans_b["choices"][0]["turns"][0].strip() if ans_b["choices"][0]["turns"] else ""
                    )
                else:
                    mds[base + 2] = "##### Assistant B\n" + post_process_answer(
                        str(ans_b["choices"][0]).strip()
                    )
            else:
                mds[base + 2] = "##### Assistant B\n" + "No answer available."

    ref = question.get("reference", [])  # Default to empty list if no reference
    if not isinstance(ref, list):
        ref = [ref]  # Convert single reference to list

    if turn is None:
        if ref:  # If there are any references
            if len(ref) == 1:
                mds[6] = f"##### Reference Solution\nQ1. {ref[0]}"
            elif len(ref) >= 2:
                mds[6] = f"##### Reference Solution\nQ1. {ref[0]}\nQ2. {ref[1]}"
    else:
        if ref and turn < len(ref) and ref[turn]:
            mds[6] = f"##### Reference Solution\n{ref[turn]}"
        else:
            mds[6] = ""

    return mds


def single_to_gradio_chat_mds(question, ans, turn=None):
    # Handle single-turn vs multi-turn questions
    if not isinstance(question.get("turns", []), list):
        question["turns"] = [question.get("turns", "")]
    
    # Ensure turns exists in question
    if "turns" not in question:
        question["turns"] = [""]
    
    end = min(len(question["turns"]), 2)  # Display up to 2 turns
    if turn is not None:
        end = min(turn + 1, end)

    mds = ["", "", "", "", ""]
    
    for i in range(end):
        base = i * 2
        if i == 0:
            mds[base + 0] = "##### User\n" + question["turns"][i]
        else:
            mds[base + 0] = "##### User's follow-up question \n" + question["turns"][i]
        
        # Handle single-turn answers
        if "choices" in ans and "turns" in ans["choices"][0]:
            turns_a = ans["choices"][0]["turns"]
            if i < len(turns_a):
                mds[base + 1] = "##### Assistant A\n" + post_process_answer(
                    turns_a[i].strip()
                )
            else:
                mds[base + 1] = "##### Assistant A\n" + "No answer for this turn."
        else:
            # Single turn format
            if i == 0 and "choices" in ans and isinstance(ans["choices"], list) and ans["choices"]:
                if isinstance(ans["choices"][0], dict) and "turns" in ans["choices"][0]:
                    mds[base + 1] = "##### Assistant A\n" + post_process_answer(
                        ans["choices"][0]["turns"][0].strip() if ans["choices"][0]["turns"] else ""
                    )
                else:
                    mds[base + 1] = "##### Assistant A\n" + post_process_answer(
                        str(ans["choices"][0]).strip()
                    )
            else:
                mds[base + 1] = "##### Assistant A\n" + "No answer available."

    ref = question.get("reference", [])  # Default to empty list if no reference
    if not isinstance(ref, list):
        ref = [ref]  # Convert single reference to list

    if turn is None:
        if ref:  # If there are any references
            if len(ref) == 1:
                mds[4] = f"##### Reference Solution\nQ1. {ref[0]}"
            elif len(ref) >= 2:
                mds[4] = f"##### Reference Solution\nQ1. {ref[0]}\nQ2. {ref[1]}"
    else:
        if ref and turn < len(ref) and ref[turn]:
            mds[4] = f"##### Reference Solution\n{ref[turn]}"
        else:
            mds[4] = ""

    return mds


def build_question_selector_map():
    global question_selector_map, category_selector_map

    # Build question selector map
    for q in questions:
        # Handle both multi-turn and single-turn formats
        if isinstance(q.get("turns", []), list) and q["turns"]:
            preview_text = q["turns"][0]
        else:
            preview_text = q.get("turns", "")
            
        preview = f"{q['question_id']}: " + str(preview_text)[:128] + "..."
        question_selector_map[preview] = q
        category_selector_map[q["category"]].append(preview)


def build_pairwise_browser_tab():
    global question_selector_map, category_selector_map

    models = list(model_answers.keys())
    num_sides = 2
    num_turns = 2
    side_names = ["A", "B"]

    question_selector_choices = list(question_selector_map.keys())
    category_selector_choices = list(category_selector_map.keys())

    # Selectors
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            category_selector = gr.Dropdown(
                choices=category_selector_choices, label="Category", container=False
            )
        with gr.Column(scale=100):
            question_selector = gr.Dropdown(
                choices=question_selector_choices, label="Question", container=False
            )

    model_selectors = [None] * num_sides
    with gr.Row():
        for i in range(num_sides):
            with gr.Column():
                if i == 0:
                    value = models[0] if models else ""
                else:
                    value = models[1] if len(models) > 1 else models[0] if models else ""
                model_selectors[i] = gr.Dropdown(
                    choices=models,
                    value=value,
                    label=f"Model {side_names[i]}",
                    container=False,
                )

    # Conversation
    chat_mds = []
    for i in range(num_turns):
        chat_mds.append(gr.Markdown(elem_id=f"user_question_{i+1}"))
        with gr.Row():
            for j in range(num_sides):
                with gr.Column(scale=100):
                    chat_mds.append(gr.Markdown())

                if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()
    reference = gr.Markdown(elem_id=f"reference")
    chat_mds.append(reference)

    model_explanation = gr.Markdown(elem_id="model_explanation")
    model_explanation2 = gr.Markdown(elem_id="model_explanation")

    # Callbacks
    category_selector.change(display_question, [category_selector], [question_selector])
    question_selector.change(
        display_pairwise_answer,
        [question_selector] + model_selectors + [gr.State(args.judge_model)],  # Add judge_model
        chat_mds + [model_explanation] + [model_explanation2],
    )

    for i in range(num_sides):
        model_selectors[i].change(
            display_pairwise_answer,
            [question_selector] + model_selectors + [gr.State(args.judge_model)],  # Add judge_model
            chat_mds + [model_explanation] + [model_explanation2],
        )

    return (category_selector,)


def build_single_answer_browser_tab():
    global question_selector_map, category_selector_map

    models = list(model_answers.keys())
    num_sides = 1
    num_turns = 2
    side_names = ["A"]

    question_selector_choices = list(question_selector_map.keys())
    category_selector_choices = list(category_selector_map.keys())

    # Selectors
    with gr.Row():
        with gr.Column(scale=1, min_width=200):
            category_selector = gr.Dropdown(
                choices=category_selector_choices, label="Category", container=False
            )
        with gr.Column(scale=100):
            question_selector = gr.Dropdown(
                choices=question_selector_choices, label="Question", container=False
            )

    model_selectors = [None] * num_sides
    with gr.Row():
        for i in range(num_sides):
            with gr.Column():
                model_selectors[i] = gr.Dropdown(
                    choices=models,
                    value=models[i] if len(models) > i else "",
                    label=f"Model {side_names[i]}",
                    container=False,
                )

    # Conversation
    chat_mds = []
    for i in range(num_turns):
        chat_mds.append(gr.Markdown(elem_id=f"user_question_{i+1}"))
        with gr.Row():
            for j in range(num_sides):
                with gr.Column(scale=100):
                    chat_mds.append(gr.Markdown())

                if j == 0:
                    with gr.Column(scale=1, min_width=8):
                        gr.Markdown()

    reference = gr.Markdown(elem_id=f"reference")
    chat_mds.append(reference)

    model_explanation = gr.Markdown(elem_id="model_explanation")
    model_explanation2 = gr.Markdown(elem_id="model_explanation")

    # Callbacks
    category_selector.change(display_question, [category_selector], [question_selector])
    question_selector.change(
        display_single_answer,
        [question_selector] + model_selectors + [gr.State(args.judge_model)],  # Add judge_model
        chat_mds + [model_explanation] + [model_explanation2],
    )

    for i in range(num_sides):
        model_selectors[i].change(
            display_single_answer,
            [question_selector] + model_selectors + [gr.State(args.judge_model)],  # Add judge_model
            chat_mds + [model_explanation] + [model_explanation2],
        )

    return (category_selector,)


block_css = """
#user_question_1 {
    background-color: rgba(222, 235, 247, 0.5);
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    color: var(--body-text-color);
}

#user_question_2 {
    background-color: rgba(226, 240, 217, 0.5);
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    color: var(--body-text-color);
}

#reference {
    background-color: rgba(255, 242, 204, 0.5);
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    color: var(--body-text-color);
}

#model_explanation {
    background-color: rgba(251, 229, 214, 0.5);
    padding: 10px;
    border-radius: 5px;
    margin: 5px 0;
    color: var(--body-text-color);
}

/* Adding styles for headers within these sections */
#user_question_1 h5,
#user_question_2 h5,
#reference h5,
#model_explanation h5 {
    color: var(--heading-text-color);
    margin-bottom: 8px;
}

/* Adding styles for the content */
#user_question_1 p,
#user_question_2 p,
#reference p,
#model_explanation p {
    color: var(--body-text-color);
    margin: 0;
}
"""


def load_demo():
    first_category = list(category_selector_map.keys())[0] if category_selector_map else None
    return first_category, first_category

def build_demo():
    build_question_selector_map()

    with gr.Blocks(
        title="MT-Bench Browser",
        theme=gr.themes.Base(text_size=gr.themes.sizes.text_lg),
        css=block_css,
    ) as demo:
        gr.Markdown(
            """
# MT-Bench Browser
The code to generate answers and judgments is at [fastchat.llm_judge](https://github.com/lm-sys/FastChat/tree/main/fastchat/llm_judge).
"""
        )
        
        category_selector = None
        category_selector2 = None
        
        # Only show tabs for which we have judgment files
        if model_judgments_normal_single is not None:
            with gr.Tab("Single Answer Grading"):
                (category_selector,) = build_single_answer_browser_tab()
                
        if model_judgments_normal_pairwise is not None:
            with gr.Tab("Pairwise Comparison"):
                (category_selector2,) = build_pairwise_browser_tab()
        
        # Handle the demo.load based on which tabs are available
        if category_selector and category_selector2:
            demo.load(load_demo, [], [category_selector, category_selector2])
        elif category_selector:
            demo.load(load_demo, [], [category_selector])
        elif category_selector2:
            demo.load(load_demo, [], [category_selector2])

    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int)
    parser.add_argument("--share", action="store_true")
    parser.add_argument("--bench-name", type=str, default="mt_bench")
    parser.add_argument("--judge-model", type=str, default="gpt-4o-mini")
    args = parser.parse_args()
    print(args)

    question_file = f"data/{args.bench_name}/question.jsonl"
    answer_dir = f"data/{args.bench_name}/model_answer"
    pairwise_model_judgment_file = (
        f"data/{args.bench_name}/model_judgment/{args.judge_model}_pair.jsonl"
    )
    single_model_judgment_file = (
        f"data/{args.bench_name}/model_judgment/{args.judge_model}_single.jsonl"
    )

    # Load questions
    questions = load_questions(question_file, None, None)

    # Load answers
    model_answers = load_model_answers(answer_dir)

    # Load model judgments
    model_judgments_normal_single = model_judgments_math_single = None
    if os.path.exists(single_model_judgment_file):
        model_judgments_normal_single = model_judgments_math_single = load_single_model_judgments(single_model_judgment_file)

    model_judgments_normal_pairwise = model_judgments_math_pairwise = None
    if os.path.exists(pairwise_model_judgment_file):
        model_judgments_normal_pairwise = model_judgments_math_pairwise = load_pairwise_model_judgments(pairwise_model_judgment_file)

    demo = build_demo()
    demo.queue(
        default_concurrency_limit=10, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )