#!/usr/bin/env bash

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
book_dir="$script_dir/book"
site_book_dir="$script_dir/_site/book"

quarto render "$book_dir"

rm -rf "$site_book_dir"
mkdir -p "$site_book_dir"
cp -R "$book_dir/_book/." "$site_book_dir"