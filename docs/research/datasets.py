"""Helpers for the interactive datasets documentation page."""

from __future__ import annotations

import base64
import html
import inspect
import io
import re
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer, make_column_selector
from sklearn.kernel_ridge import KernelRidge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from skcausal.causal_estimators import DirectNoCovariates
from skcausal.datatypes import collect_column_types, convert
from skcausal.plotting import plot_marginal_curves
from skcausal.utils.lookup import all_datasets

_INLINE_RST_PATTERN = re.compile(
    r":(?P<role>math|class|meth):`(?P<role_content>[^`]+)`|``(?P<literal>[^`]+)``"
)
_SECTION_UNDERLINE_PATTERN = re.compile(r"^[-=~`:#\"'^*]{3,}$")
_COLON_FIELD_PATTERN = re.compile(
    r"^\s*(?P<name>[^:\n][^:\n]*?)\s*:\s*(?P<annotation>.+?)\s*$"
)
_COLON_FIELD_SECTIONS = {"Parameters", "Other Parameters", "Attributes"}
_TYPE_ONLY_FIELD_SECTIONS = {"Returns", "Yields", "Raises"}


def _coerce_curve(values: Any) -> np.ndarray:
    if isinstance(values, pl.DataFrame):
        if values.width != 1:
            raise ValueError("Expected a single-column frame for curve values.")
        values = values.to_numpy()

    array = np.asarray(values, dtype=float)
    if array.ndim == 0:
        raise ValueError("Expected one value per treatment row.")

    array = array.reshape(array.shape[0], -1)
    if array.shape[1] != 1:
        raise ValueError(f"Expected a single-output curve, got shape {array.shape}.")

    return array[:, 0]


def _get_docstring(dataset_cls: type) -> str:
    return inspect.getdoc(dataset_cls) or ""


def _is_section_heading(lines: list[str], index: int) -> bool:
    if index + 1 >= len(lines):
        return False

    title = lines[index].strip()
    underline = lines[index + 1].strip()
    return bool(title) and bool(_SECTION_UNDERLINE_PATTERN.fullmatch(underline))


def _normalize_rst_role_label(content: str) -> str:
    label = content.strip().lstrip("~")
    if " <" in label and label.endswith(">"):
        label = label.split("<", maxsplit=1)[0].strip()
    return label.rsplit(".", maxsplit=1)[-1]


def _leading_indent(line: str) -> int:
    return len(line) - len(line.lstrip(" \t"))


def _render_inline_rst(text: str) -> str:
    rendered: list[str] = []
    last_end = 0

    for match in _INLINE_RST_PATTERN.finditer(text):
        rendered.append(html.escape(text[last_end : match.start()]))
        last_end = match.end()

        literal = match.group("literal")
        if literal is not None:
            rendered.append(f"<code>{html.escape(literal)}</code>")
            continue

        role = match.group("role")
        role_content = match.group("role_content")
        if role == "math":
            rendered.append(rf"\({role_content}\)")
        else:
            rendered.append(
                f"<code>{html.escape(_normalize_rst_role_label(role_content))}</code>"
            )

    rendered.append(html.escape(text[last_end:]))
    return "".join(rendered)


def _looks_like_type_only_field_header(lines: list[str], index: int) -> bool:
    stripped = lines[index].strip()
    if not stripped or ":" in stripped:
        return False

    header_indent = _leading_indent(lines[index])
    probe = index + 1
    while probe < len(lines) and not lines[probe].strip():
        probe += 1

    if probe >= len(lines):
        return False
    if _is_section_heading(lines, probe) or lines[probe].strip() == ".. math::":
        return False

    return _leading_indent(lines[probe]) > header_indent


def _looks_like_colon_field_header(lines: list[str], index: int) -> bool:
    if _COLON_FIELD_PATTERN.match(lines[index]) is None:
        return False

    header_indent = _leading_indent(lines[index])
    probe = index + 1
    while probe < len(lines) and not lines[probe].strip():
        probe += 1

    if probe >= len(lines):
        return True
    if _is_section_heading(lines, probe) or lines[probe].strip() == ".. math::":
        return True

    return _leading_indent(lines[probe]) > header_indent


def _split_paragraphs(lines: list[str]) -> list[str]:
    paragraphs: list[str] = []
    current: list[str] = []

    for line in lines:
        if not line:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        current.append(line)

    if current:
        paragraphs.append(" ".join(current))

    return paragraphs


def _render_field_list_html(section_title: str, items: list[dict[str, Any]]) -> str:
    section_slug = _slugify(section_title)
    blocks = [
        '<dl class="dataset-docstring__fields '
        f'dataset-docstring__fields--{section_slug}">'
    ]

    for item in items:
        term_parts: list[str] = []
        if item["name"]:
            term_parts.append(
                '<code class="dataset-docstring__field-name">'
                f'{html.escape(item["name"])}'
                "</code>"
            )
        if item["annotation"]:
            if item["name"]:
                term_parts.append(
                    '<span class="dataset-docstring__field-separator">:</span>'
                )
            term_parts.append(
                '<span class="dataset-docstring__field-type">'
                f'{_render_inline_rst(item["annotation"])}'
                "</span>"
            )

        blocks.append(
            '<dt class="dataset-docstring__field-term">'
            f'{"".join(term_parts)}'
            "</dt>"
        )

        description_html = "".join(
            '<p class="dataset-docstring__field-paragraph">'
            f"{_render_inline_rst(paragraph)}"
            "</p>"
            for paragraph in _split_paragraphs(item["description_lines"])
        )
        blocks.append(
            '<dd class="dataset-docstring__field-body">' f"{description_html}" "</dd>"
        )

    blocks.append("</dl>")
    return "\n".join(blocks)


def _parse_field_list(
    lines: list[str],
    index: int,
    *,
    section_title: str,
    allow_type_only: bool,
) -> tuple[str, int] | None:
    items: list[dict[str, Any]] = []
    cursor = index

    while cursor < len(lines):
        while cursor < len(lines) and not lines[cursor].strip():
            cursor += 1

        if cursor >= len(lines):
            break
        if _is_section_heading(lines, cursor) or lines[cursor].strip() == ".. math::":
            break

        header_line = lines[cursor]
        header_indent = _leading_indent(header_line)
        field_match = _COLON_FIELD_PATTERN.match(header_line)

        if field_match is not None and _looks_like_colon_field_header(lines, cursor):
            field_name = field_match.group("name").strip()
            field_annotation = field_match.group("annotation").strip()
        elif allow_type_only and _looks_like_type_only_field_header(lines, cursor):
            field_name = None
            field_annotation = header_line.strip()
        else:
            return None if not items else ("", index)

        cursor += 1
        description_lines: list[str] = []

        while cursor < len(lines):
            raw_line = lines[cursor]
            stripped = raw_line.strip()

            if not stripped:
                if description_lines and description_lines[-1] != "":
                    description_lines.append("")
                cursor += 1
                continue

            if _is_section_heading(lines, cursor) or stripped == ".. math::":
                break

            if _looks_like_colon_field_header(lines, cursor):
                break

            if (
                allow_type_only
                and _looks_like_type_only_field_header(lines, cursor)
                and _leading_indent(raw_line) <= header_indent
            ):
                break

            description_lines.append(stripped)
            cursor += 1

        items.append(
            {
                "name": field_name,
                "annotation": field_annotation,
                "description_lines": description_lines,
            }
        )

    if not items:
        return None

    return _render_field_list_html(section_title, items), cursor


def _render_docstring_html(doc: str) -> str:
    if not doc:
        return ""

    blocks: list[str] = []
    lines = doc.splitlines()
    index = 0
    current_section: str | None = None

    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue

        if stripped == ".. math::":
            index += 1
            while index < len(lines) and not lines[index].strip():
                index += 1

            math_lines: list[str] = []
            math_indent: int | None = None
            while index < len(lines):
                raw_line = lines[index]
                if not raw_line.strip():
                    if math_indent is not None:
                        math_lines.append("")
                    index += 1
                    continue

                current_indent = _leading_indent(raw_line)
                if current_indent == 0:
                    break
                if math_indent is None:
                    math_indent = current_indent
                if current_indent < math_indent:
                    break

                math_lines.append(raw_line[math_indent:])
                index += 1

            math_body = "\n".join(math_lines).strip()
            if math_body:
                blocks.append(
                    '<div class="dataset-docstring__math">$$\n'
                    f"{math_body}\n"
                    "$$</div>"
                )
            current_section = None
            continue

        if _is_section_heading(lines, index):
            blocks.append(
                '<h4 class="dataset-docstring__heading">'
                f"{html.escape(stripped)}"
                "</h4>"
            )
            current_section = stripped
            index += 2
            continue

        if current_section in _COLON_FIELD_SECTIONS:
            parsed_field_list = _parse_field_list(
                lines,
                index,
                section_title=current_section,
                allow_type_only=False,
            )
            if parsed_field_list is not None:
                field_list_html, next_index = parsed_field_list
                blocks.append(field_list_html)
                index = next_index
                continue

        if current_section in _TYPE_ONLY_FIELD_SECTIONS:
            parsed_field_list = _parse_field_list(
                lines,
                index,
                section_title=current_section,
                allow_type_only=True,
            )
            if parsed_field_list is not None:
                field_list_html, next_index = parsed_field_list
                blocks.append(field_list_html)
                index = next_index
                continue

        paragraph_lines = [stripped]
        index += 1
        while index < len(lines):
            next_line = lines[index]
            next_stripped = next_line.strip()
            if not next_stripped:
                index += 1
                break
            if next_stripped == ".. math::" or _is_section_heading(lines, index):
                break
            paragraph_lines.append(next_stripped)
            index += 1

        paragraph = " ".join(paragraph_lines)
        blocks.append(
            '<p class="dataset-docstring__paragraph">'
            f"{_render_inline_rst(paragraph)}"
            "</p>"
        )

    return "\n".join(blocks)


def _format_params(params: dict[str, Any]) -> str:
    return ", ".join(
        f"{key}={value!r}"
        for key, value in sorted(params.items(), key=lambda item: item[0])
    )


def _slugify(value: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.lower()).strip("-")
    return slug or "dataset"


def _figure_to_data_uri(fig) -> str:
    buffer = io.BytesIO()
    fig.savefig(buffer, format="png", dpi=160, bbox_inches="tight")
    buffer.seek(0)
    encoded = base64.b64encode(buffer.read()).decode("ascii")
    return f"data:image/png;base64,{encoded}"


def _get_plot_treatments(treatments: pl.DataFrame) -> pl.DataFrame:
    return treatments


def _make_naive_estimator() -> DirectNoCovariates:
    preprocessor = ColumnTransformer(
        transformers=[
            (
                "encode_categorical",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                make_column_selector(
                    dtype_include=["category", "object", "string", "bool"]
                ),
            ),
            (
                "scale_numeric",
                StandardScaler(),
                make_column_selector(dtype_include=np.number),
            ),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )
    outcome_regressor = make_pipeline(preprocessor, KernelRidge(kernel="rbf"))
    return DirectNoCovariates(outcome_regressor=outcome_regressor)


def _fit_naive_curve(
    covariates: pl.DataFrame,
    treatments: pl.DataFrame,
    outcomes: pl.DataFrame,
    plot_treatments: pl.DataFrame,
) -> np.ndarray:
    naive = _make_naive_estimator()
    naive.fit(covariates, treatments, outcomes)
    return _coerce_curve(naive.predict(plot_treatments))


def _build_example(instance_name: str, dataset) -> dict[str, Any]:
    covariates, treatments, outcomes = dataset.load()
    plot_treatments = _get_plot_treatments(treatments)

    curves = {
        "Groundtruth": _coerce_curve(
            dataset.predict_curve(covariates=covariates, treatment_grid=plot_treatments)
        ),
        "Naive": _fit_naive_curve(
            covariates=covariates,
            treatments=treatments,
            outcomes=outcomes,
            plot_treatments=plot_treatments,
        ),
    }
    if plot_treatments.height == outcomes.height:
        curves = {
            "Observed outcome": _coerce_curve(outcomes),
            **curves,
        }

    return {
        "example_slug": _slugify(instance_name),
        "instance_name": instance_name,
        "dataset": dataset,
        "n_rows": len(treatments),
        "treatment_columns": list(treatments.columns),
        "params": dataset.get_params(),
        "curves": curves,
        "treatments": treatments,
        "plot_treatments": plot_treatments,
    }


def iter_dataset_sections() -> list[dict[str, Any]]:
    dataset_frame = all_datasets(as_dataframe=True)
    dataset_frame = dataset_frame.sort_values("name", kind="stable").reset_index(
        drop=True
    )

    sections = []
    for _, row in dataset_frame.iterrows():
        dataset_cls = row["object"]
        instances, instance_names = dataset_cls.create_test_instances_and_names()
        first_example = next(zip(instances, instance_names), None)
        sections.append(
            {
                "dataset_name": row["name"],
                "dataset_slug": _slugify(row["name"]),
                "description_html": _render_docstring_html(
                    _get_docstring(dataset_cls).split("\n\n", maxsplit=1)[0].strip()
                ),
                "docstring_html": _render_docstring_html(_get_docstring(dataset_cls)),
                "examples": (
                    []
                    if first_example is None
                    else [_build_example(first_example[1], first_example[0])]
                ),
            }
        )

    return sections


def _refresh_legend(axis) -> None:
    handles, labels = axis.get_legend_handles_labels()
    unique_handles = []
    unique_labels = []
    seen = set()

    for handle, label in zip(handles, labels):
        if not label or label == "_nolegend_" or label in seen:
            continue
        seen.add(label)
        unique_handles.append(handle)
        unique_labels.append(label)

    if unique_handles:
        axis.legend(unique_handles, unique_labels)


def _style_observed_outcome_as_scatter(axes, treatments, curves: dict[str, np.ndarray]):
    observed = curves.get("Observed outcome")
    if observed is None:
        return

    treatment_frame = convert(treatments, "pandas")
    column_types = collect_column_types(treatment_frame)
    axes = np.atleast_1d(np.asarray(axes, dtype=object)).reshape(-1)
    rng = np.random.default_rng(0)

    for axis_index, (axis, column) in enumerate(zip(axes, treatment_frame.columns)):
        for line in list(axis.lines):
            if line.get_label() == "Observed outcome":
                line.remove()

        for container in list(axis.containers):
            if container.get_label() != "Observed outcome":
                continue
            for patch in container.patches:
                patch.remove()
            container.set_label("_nolegend_")

        scatter_label = "Observed outcome" if axis_index == 0 else None

        if column_types[column] == "continuous":
            plot_frame = treatment_frame[[column]].copy()
            plot_frame["__value__"] = np.asarray(observed, dtype=float)
            plot_frame = (
                plot_frame.groupby([column], observed=True, sort=False)["__value__"]
                .mean()
                .reset_index()
                .sort_values(column, kind="stable")
            )

            axis.scatter(
                plot_frame[column].to_numpy(dtype=float),
                plot_frame["__value__"].to_numpy(dtype=float),
                label=scatter_label,
                alpha=0.2,
                s=24,
            )
            continue

        category_series = treatment_frame[column]
        category_labels = [str(value) for value in category_series.astype(str)]
        if hasattr(category_series, "cat"):
            observed_labels = set(category_labels)
            unique_labels = [
                str(category)
                for category in category_series.cat.categories
                if str(category) in observed_labels
            ]
        else:
            unique_labels = list(dict.fromkeys(category_labels))

        naive_container = next(
            (
                container
                for container in axis.containers
                if container.get_label() == "Naive"
            ),
            None,
        )

        jitter_scale = 0.12
        if naive_container is not None and len(naive_container.patches) == len(
            unique_labels
        ):
            category_positions = {
                label: patch.get_x() + patch.get_width() / 2.0
                for label, patch in zip(unique_labels, naive_container.patches)
            }
            jitter_scale = naive_container.patches[0].get_width() * 0.35
        else:
            category_positions = {
                label: float(index) for index, label in enumerate(unique_labels)
            }

        x_positions = np.asarray(
            [category_positions[label] for label in category_labels],
            dtype=float,
        )
        x_positions = x_positions + rng.uniform(
            -jitter_scale,
            jitter_scale,
            size=len(x_positions),
        )

        axis.scatter(
            x_positions,
            np.asarray(observed, dtype=float),
            label=scatter_label,
            alpha=0.18,
            s=18,
            zorder=3,
        )

    if len(curves) > 1:
        _refresh_legend(axes[0])


def make_dataset_figure(example: dict[str, Any]):
    n_axes = len(example["treatment_columns"])
    fig, axes = plt.subplots(
        1,
        n_axes,
        figsize=(max(6.0, 4.8 * n_axes), 4.2),
        squeeze=False,
        sharey=True,
    )
    axes = axes[0]

    plot_marginal_curves(
        example["plot_treatments"],
        example["curves"],
        ax=axes[0] if n_axes == 1 else axes,
    )
    _style_observed_outcome_as_scatter(
        axes,
        example["plot_treatments"],
        example["curves"],
    )
    fig.suptitle(example["instance_name"], fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.94))
    return fig


def format_example_summary(example: dict[str, Any]) -> str:
    treatment_columns = ", ".join(example["treatment_columns"])
    parameter_summary = _format_params(example["params"])
    return (
        f"Rows: {example['n_rows']}. "
        f"Treatment columns: {treatment_columns}. "
        f"Parameters: {parameter_summary}."
    )


def render_dataset_browser_html(sections: list[dict[str, Any]] | None = None) -> str:
    sections = iter_dataset_sections() if sections is None else sections
    if not sections:
        return (
            '<div class="dataset-browser"><p>No datasets were discovered by '
            "the registry.</p></div>"
        )

    option_lines = []
    panel_lines = []

    for index, section in enumerate(sections):
        dataset_name = html.escape(section["dataset_name"])
        dataset_slug = section["dataset_slug"]
        selected = " selected" if index == 0 else ""
        hidden = "" if index == 0 else " hidden"

        option_lines.append(
            f'<option value="{dataset_slug}"{selected}>{dataset_name}</option>'
        )

        panel_lines.append(
            f'<section class="dataset-panel" data-dataset-panel="{dataset_slug}"{hidden}>'
        )
        panel_lines.append(f'<h2 class="dataset-panel__title">{dataset_name}</h2>')

        if section["description_html"]:
            panel_lines.append(
                '<div class="dataset-panel__description">'
                f'{section["description_html"]}'
                "</div>"
            )

        for example in section["examples"]:
            figure = make_dataset_figure(example)
            image_src = _figure_to_data_uri(figure)
            plt.close(figure)

            panel_lines.append('<article class="dataset-example">')
            panel_lines.append(
                '<div class="dataset-example__meta">'
                f'<h3 class="dataset-example__title">{html.escape(example["instance_name"])}</h3>'
                f'<p class="dataset-example__summary">{html.escape(format_example_summary(example))}</p>'
                "</div>"
            )
            panel_lines.append(
                '<img class="dataset-example__image" '
                f'src="{image_src}" '
                f'alt="Treatment versus response plot for {html.escape(example["instance_name"])}">'
            )
            if section["docstring_html"]:
                panel_lines.append(
                    '<div class="dataset-example__docstring">'
                    f'{section["docstring_html"]}'
                    "</div>"
                )
            panel_lines.append("</article>")

        panel_lines.append("</section>")

    return "\n".join(
        [
            '<div class="dataset-browser">',
            '<div class="dataset-browser__controls">',
            '<label class="dataset-browser__label" for="dataset-browser-select">Choose dataset</label>',
            '<select class="dataset-browser__select" id="dataset-browser-select">',
            *option_lines,
            "</select>",
            "</div>",
            '<div class="dataset-browser__panels">',
            *panel_lines,
            "</div>",
            "</div>",
        ]
    )
