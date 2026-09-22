#!/usr/bin/env python3
"""Regenerate full-denominator relation and length diagnostics for the paper.

The script treats the gpt-oss Zero-Shot evaluation as the complete 2,077-sentence
gold reference. Every model/condition is scored against that reference, so a
missing record, schema failure, or missing token contributes zero correct arcs
without reducing a relation's gold support.

Outputs
-------
results/per_relation.json
figures/fig_per_relation_selected.png
figures/fig_per_relation_selected.pdf
results/length_analysis.json
figures/fig_las_by_length_cross_model.png
figures/fig_las_by_length_cross_model.pdf
"""

from __future__ import annotations

import json
import gzip
from collections import defaultdict
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
REFERENCE = ROOT / "outputs/gpt_oss_120b/zero_shot/model_evaluation.json"
JSON_OUT = ROOT / "results/per_relation.json"
PNG_OUT = ROOT / "figures/fig_per_relation_selected.png"
PDF_OUT = ROOT / "figures/fig_per_relation_selected.pdf"
LENGTH_JSON_OUT = ROOT / "results/length_analysis.json"
LENGTH_PNG_OUT = ROOT / "figures/fig_las_by_length_cross_model.png"
LENGTH_PDF_OUT = ROOT / "figures/fig_las_by_length_cross_model.pdf"

MODELS = {
    "gpt_oss_120b": "gpt-oss-120b",
    "qwen25_72b": "Qwen2.5-72B",
}

CONDITIONS = [
    ("zero_shot", "Zero-Shot", "#999999"),
    ("few_shot_fixed", "Fixed Few-Shot", "#E6A23C"),
    ("few_shot_retrieval", "POS-Signature", "#2F9E73"),
    ("semantic_retrieval", "Semantic", "#8B6AAF"),
    ("cot_few_shot", "CoT", "#3977B7"),
    ("critique_refine", "Critique-Refine", "#D35F32"),
]

FIGURE_RELATIONS = [
    "nsubj",
    "nmod:poss",
    "cc",
    "cop",
    "nmod:unmarked",
    "obl:unmarked",
]

LENGTH_BINS = [
    ("1–10", 1, 10),
    ("11–20", 11, 20),
    ("21–30", 21, 30),
    ("31–50", 31, 50),
    ("51+", 51, 10**9),
]


def load_json(path: Path):
    if not path.exists():
        path = Path(str(path) + ".gz")
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def score_relations() -> dict:
    reference = load_json(REFERENCE)
    support: dict[str, int] = defaultdict(int)
    for record in reference:
        for gold in record["gold"]:
            if gold["upos"] != "PUNCT":
                support[gold["deprel"]] += 1

    result = {
        "protocol": {
            "reference_sentences": len(reference),
            "reference_non_punctuation_tokens": sum(support.values()),
            "missing_or_unscorable_policy": "zero correct; gold support retained",
            "relation_subtypes": "retained",
        },
        "support": dict(sorted(support.items(), key=lambda item: (-item[1], item[0]))),
        "models": {},
    }

    for model_slug in MODELS:
        model_result = {}
        for condition, _, _ in CONDITIONS:
            path = ROOT / f"outputs/{model_slug}/{condition}/model_evaluation.json"
            records = {record["sent_id"]: record for record in load_json(path)}
            las_correct: dict[str, int] = defaultdict(int)

            for gold_record in reference:
                prediction_record = records.get(gold_record["sent_id"])
                prediction = (
                    prediction_record.get("llm_parse")
                    if prediction_record is not None
                    else None
                )
                pred_by_id = {
                    token["id"]: token for token in prediction or []
                }

                for gold in gold_record["gold"]:
                    if gold["upos"] == "PUNCT":
                        continue
                    predicted = pred_by_id.get(gold["id"])
                    if (
                        predicted is not None
                        and predicted["head"] == gold["head"]
                        and predicted["deprel"].lower() == gold["deprel"].lower()
                    ):
                        las_correct[gold["deprel"]] += 1

            condition_result = {}
            for relation, total in support.items():
                correct = las_correct[relation]
                condition_result[relation] = {
                    "support": total,
                    "las_correct": correct,
                    "las": correct / total,
                }
            model_result[condition] = condition_result
        result["models"][model_slug] = model_result

    return result


def score_lengths() -> dict:
    reference = load_json(REFERENCE)
    zero_token_sentences = sum(
        1
        for record in reference
        if not any(gold["upos"] != "PUNCT" for gold in record["gold"])
    )
    result = {
        "protocol": {
            "reference_sentences": len(reference),
            "reference_non_punctuation_tokens": sum(
                1
                for record in reference
                for gold in record["gold"]
                if gold["upos"] != "PUNCT"
            ),
            "length_definition": "number of gold non-PUNCT tokens",
            "zero_scored_token_sentences_excluded_from_bins": zero_token_sentences,
            "missing_or_unscorable_policy": "zero correct; sentence and gold tokens retained",
            "status": "post-hoc descriptive analysis",
        },
        "models": {},
    }

    for model_slug in MODELS:
        model_result = {}
        for condition, _, _ in CONDITIONS:
            path = ROOT / f"outputs/{model_slug}/{condition}/model_evaluation.json"
            records = {record["sent_id"]: record for record in load_json(path)}
            bins = {
                label: {
                    "sentences": 0,
                    "total_tokens": 0,
                    "las_correct": 0,
                }
                for label, _, _ in LENGTH_BINS
            }

            for gold_record in reference:
                non_punct = [
                    gold for gold in gold_record["gold"] if gold["upos"] != "PUNCT"
                ]
                length = len(non_punct)
                if length == 0:
                    # These sentences contribute no token to LAS and therefore
                    # have no defined bin-level score.
                    continue
                label = next(
                    label
                    for label, lower, upper in LENGTH_BINS
                    if lower <= length <= upper
                )
                bucket = bins[label]
                bucket["sentences"] += 1
                bucket["total_tokens"] += length

                prediction_record = records.get(gold_record["sent_id"])
                prediction = (
                    prediction_record.get("llm_parse")
                    if prediction_record is not None
                    else None
                )
                pred_by_id = {
                    token["id"]: token for token in prediction or []
                }
                for gold in non_punct:
                    predicted = pred_by_id.get(gold["id"])
                    if (
                        predicted is not None
                        and predicted["head"] == gold["head"]
                        and predicted["deprel"].lower() == gold["deprel"].lower()
                    ):
                        bucket["las_correct"] += 1

            for bucket in bins.values():
                bucket["las"] = (
                    bucket["las_correct"] / bucket["total_tokens"]
                    if bucket["total_tokens"]
                    else None
                )
            model_result[condition] = bins
        result["models"][model_slug] = model_result

    return result


def load_font(size: int, bold: bool = False):
    candidates = [
        Path(
            "/System/Library/Fonts/Supplemental/"
            + ("Arial Bold.ttf" if bold else "Arial.ttf")
        ),
        Path(
            "/usr/share/fonts/truetype/dejavu/"
            + ("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf")
        ),
    ]
    for candidate in candidates:
        if candidate.exists():
            return ImageFont.truetype(str(candidate), size)
    return ImageFont.load_default()


def draw_centered(draw, xy, text, font, fill="#111111"):
    draw.text(xy, text, font=font, fill=fill, anchor="mm", align="center")


def render_figure(result: dict):
    width, height = 2200, 880
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_legend = load_font(27)
    font_title = load_font(35)
    font_axis = load_font(27)
    font_tick = load_font(23)
    font_rel = load_font(21)
    font_support = load_font(18)

    legend_y = 38
    legend_widths = []
    for _, label, _ in CONDITIONS:
        box = draw.textbbox((0, 0), label, font=font_legend)
        legend_widths.append(38 + box[2] - box[0] + 32)
    legend_total = sum(legend_widths)
    legend_x = (width - legend_total) / 2
    for (_, label, color), item_width in zip(CONDITIONS, legend_widths):
        draw.rectangle(
            (legend_x, legend_y - 13, legend_x + 30, legend_y + 13),
            fill=color,
            outline="#222222",
            width=1,
        )
        draw.text(
            (legend_x + 39, legend_y),
            label,
            font=font_legend,
            fill="#111111",
            anchor="lm",
        )
        legend_x += item_width

    plot_top, plot_bottom = 145, 700
    panel_lefts = [105, 1135]
    panel_width = 955
    plot_width = 900
    support = result["support"]

    for panel_index, (model_slug, model_title) in enumerate(MODELS.items()):
        left = panel_lefts[panel_index]
        right = left + plot_width
        draw_centered(
            draw,
            ((left + right) / 2, 105),
            model_title,
            font_title,
        )

        for tick in range(0, 101, 20):
            y = plot_bottom - (plot_bottom - plot_top) * tick / 100
            draw.line((left, y, right, y), fill="#D9D9D9", width=2)
            if panel_index == 0:
                draw.text(
                    (left - 20, y),
                    str(tick),
                    font=font_tick,
                    fill="#111111",
                    anchor="rm",
                )

        draw.line((left, plot_top, left, plot_bottom), fill="#111111", width=3)
        draw.line((left, plot_bottom, right, plot_bottom), fill="#111111", width=3)

        group_width = plot_width / len(FIGURE_RELATIONS)
        bar_gap = 2
        bar_width = 20
        bars_width = len(CONDITIONS) * bar_width + (len(CONDITIONS) - 1) * bar_gap

        for relation_index, relation in enumerate(FIGURE_RELATIONS):
            group_center = left + group_width * (relation_index + 0.5)
            bar_left = group_center - bars_width / 2

            for condition_index, (condition, _, color) in enumerate(CONDITIONS):
                score = (
                    result["models"][model_slug][condition][relation]["las"] * 100
                )
                x0 = bar_left + condition_index * (bar_width + bar_gap)
                x1 = x0 + bar_width
                y0 = plot_bottom - (plot_bottom - plot_top) * score / 100
                draw.rectangle(
                    (round(x0), round(y0), round(x1), plot_bottom),
                    fill=color,
                    outline="#222222",
                    width=1,
                )

            draw_centered(
                draw,
                (group_center, 742),
                relation,
                font_rel,
            )
            draw_centered(
                draw,
                (group_center, 774),
                f"n={support[relation]:,}",
                font_support,
                fill="#444444",
            )

    label_layer = Image.new("RGBA", (80, 210), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text(
        (40, 105),
        "LAS (%)",
        font=font_axis,
        fill="#111111",
        anchor="mm",
    )
    label_layer = label_layer.rotate(90, expand=True)
    image.paste(label_layer, (-75, 300), label_layer)

    PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(PNG_OUT, dpi=(220, 220), optimize=True)
    image.save(PDF_OUT, "PDF", resolution=220.0)


def render_length_figure(result: dict):
    width, height = 2200, 880
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)

    font_legend = load_font(27)
    font_title = load_font(35)
    font_axis = load_font(27)
    font_tick = load_font(23)
    font_bin = load_font(21)
    font_support = load_font(18)

    legend_y = 38
    legend_widths = []
    for _, label, _ in CONDITIONS:
        box = draw.textbbox((0, 0), label, font=font_legend)
        legend_widths.append(38 + box[2] - box[0] + 32)
    legend_x = (width - sum(legend_widths)) / 2
    for (_, label, color), item_width in zip(CONDITIONS, legend_widths):
        draw.line(
            (legend_x, legend_y, legend_x + 32, legend_y),
            fill=color,
            width=5,
        )
        draw.ellipse(
            (legend_x + 11, legend_y - 6, legend_x + 23, legend_y + 6),
            fill=color,
        )
        draw.text(
            (legend_x + 42, legend_y),
            label,
            font=font_legend,
            fill="#111111",
            anchor="lm",
        )
        legend_x += item_width

    plot_top, plot_bottom = 145, 700
    panel_lefts = [105, 1135]
    plot_width = 900
    y_min, y_max = 35, 85

    reference_model = next(iter(MODELS))
    support_by_bin = {
        label: result["models"][reference_model]["zero_shot"][label]["sentences"]
        for label, _, _ in LENGTH_BINS
    }

    for panel_index, (model_slug, model_title) in enumerate(MODELS.items()):
        left = panel_lefts[panel_index]
        right = left + plot_width
        draw_centered(
            draw,
            ((left + right) / 2, 105),
            model_title,
            font_title,
        )

        for tick in range(y_min, y_max + 1, 5):
            y = plot_bottom - (plot_bottom - plot_top) * (tick - y_min) / (
                y_max - y_min
            )
            draw.line((left, y, right, y), fill="#D9D9D9", width=2)
            if panel_index == 0:
                draw.text(
                    (left - 20, y),
                    str(tick),
                    font=font_tick,
                    fill="#111111",
                    anchor="rm",
                )

        draw.line((left, plot_top, left, plot_bottom), fill="#111111", width=3)
        draw.line((left, plot_bottom, right, plot_bottom), fill="#111111", width=3)

        x_positions = [
            left + plot_width * index / (len(LENGTH_BINS) - 1)
            for index in range(len(LENGTH_BINS))
        ]
        for condition, _, color in CONDITIONS:
            points = []
            for x, (label, _, _) in zip(x_positions, LENGTH_BINS):
                score = result["models"][model_slug][condition][label]["las"] * 100
                y = plot_bottom - (plot_bottom - plot_top) * (score - y_min) / (
                    y_max - y_min
                )
                points.append((x, y))
            draw.line(points, fill=color, width=5, joint="curve")
            for x, y in points:
                draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=color)

        for x, (label, _, _) in zip(x_positions, LENGTH_BINS):
            draw_centered(draw, (x, 742), label, font_bin)
            draw_centered(
                draw,
                (x, 774),
                f"n={support_by_bin[label]:,}",
                font_support,
                fill="#444444",
            )

    label_layer = Image.new("RGBA", (80, 210), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label_layer)
    label_draw.text(
        (40, 105),
        "LAS (%)",
        font=font_axis,
        fill="#111111",
        anchor="mm",
    )
    label_layer = label_layer.rotate(90, expand=True)
    image.paste(label_layer, (-75, 300), label_layer)

    LENGTH_PNG_OUT.parent.mkdir(parents=True, exist_ok=True)
    LENGTH_PDF_OUT.parent.mkdir(parents=True, exist_ok=True)
    image.save(LENGTH_PNG_OUT, dpi=(220, 220), optimize=True)
    image.save(LENGTH_PDF_OUT, "PDF", resolution=220.0)


def main():
    result = score_relations()
    length_result = score_lengths()
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with JSON_OUT.open("w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    with LENGTH_JSON_OUT.open("w", encoding="utf-8") as handle:
        json.dump(length_result, handle, indent=2, ensure_ascii=False)
        handle.write("\n")
    render_figure(result)
    render_length_figure(length_result)

    for relation in ["nmod:poss", "cop", "nmod:unmarked"]:
        row = {"relation": relation, "support": result["support"][relation]}
        for model_slug in MODELS:
            for condition in ["semantic_retrieval", "few_shot_retrieval"]:
                key = f"{model_slug}:{condition}"
                row[key] = round(
                    100 * result["models"][model_slug][condition][relation]["las"],
                    2,
                )
        print(json.dumps(row, ensure_ascii=False))


if __name__ == "__main__":
    main()
