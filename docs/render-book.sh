#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
book_dir="$script_dir/book"
book_output_dir="$book_dir/_book"
site_book_dir="$script_dir/_site/book"
book_render_stamp="$book_output_dir/.render-stamp"

book_sources_changed() {
	if [[ ! -d "$book_output_dir" || ! -f "$book_render_stamp" ]]; then
		return 0
	fi

	find "$book_dir" \
		-type f \
		\( \
			-name '*.qmd' -o \
			-name '*.md' -o \
			-name '*.yml' -o \
			-name '*.yaml' -o \
			-name '*.css' -o \
			-name '*.ipynb' -o \
			-name '*.png' -o \
			-name '*.jpg' -o \
			-name '*.jpeg' -o \
			-name '*.svg' \
		\) \
		! -path "$book_output_dir/*" \
		-newer "$book_render_stamp" \
		-print -quit | grep -q .
}

if book_sources_changed; then
	echo "Rendering nested book..."
	quarto render "$book_dir"
	mkdir -p "$book_output_dir"
	touch "$book_render_stamp"
else
	echo "Reusing existing nested book build."
fi

if [[ ! -d "$book_output_dir" ]]; then
	echo "Expected rendered book output at $book_output_dir." >&2
	exit 1
fi

rm -rf "$site_book_dir"
mkdir -p "$site_book_dir"
cp -R "$book_output_dir/." "$site_book_dir"