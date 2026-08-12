#!/bin/bash

# Default: include examples
INCLUDE_EXAMPLES=true
SKIP_COMMENTS=false
SKIP_TESTS=false
MAX_LINES=20000
IGNORE_DIRS=()

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --no-examples)
            INCLUDE_EXAMPLES=false
            shift
            ;;
        --no-comments)
            SKIP_COMMENTS=true
            shift
            ;;
        --no-tests)
            SKIP_TESTS=true
            shift
            ;;
        --max-lines)
            MAX_LINES="$2"
            shift 2
            ;;
        --ignore-dir)
            IGNORE_DIRS+=("$2")
            shift 2
            ;;
        -h|--help)
            echo "Usage: $0 [--no-examples] [--no-comments] [--no-tests] [--max-lines N] [--ignore-dir DIR]"
            echo "  --no-examples    Exclude examples directory from collection"
            echo "  --no-comments    Skip comment lines from source files"
            echo "  --no-tests       Skip Rust test modules and functions"
            echo "  --max-lines N    Split output into multiple files after N lines (default: 10000)"
            echo "  --ignore-dir DIR Ignore a specific directory (can be used multiple times)"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# Output file management
OUTPUT_BASE="all_sources_pp_ex"
FILE_NUM=1
CURRENT_LINE_COUNT=0
OUTPUT_FILE="${OUTPUT_BASE}.md"

# Function to write header to a new file
write_header() {
    local file="$1"
    local part_num="$2"
    
    if [[ $part_num -gt 1 ]]; then
        echo "# Source Code (Part $part_num)" > "$file"
    else
        echo "# Source Code" > "$file"
    fi
    echo "" >> "$file"
    echo "Generated on: $(date)" >> "$file"
    if [[ "$INCLUDE_EXAMPLES" == false ]] || [[ "$SKIP_COMMENTS" == true ]] || [[ "$SKIP_TESTS" == true ]]; then
        echo -n "(" >> "$file"
        SEPARATOR=""
        if [[ "$INCLUDE_EXAMPLES" == false ]]; then
            echo -n "${SEPARATOR}Examples excluded" >> "$file"
            SEPARATOR=", "
        fi
        if [[ "$SKIP_COMMENTS" == true ]]; then
            echo -n "${SEPARATOR}Comments excluded" >> "$file"
            SEPARATOR=", "
        fi
        if [[ "$SKIP_TESTS" == true ]]; then
            echo -n "${SEPARATOR}Tests excluded" >> "$file"
        fi
        echo ")" >> "$file"
    fi
    echo "" >> "$file"
    
    # Return the number of lines written
    wc -l < "$file"
}

# Function to start a new output file
start_new_file() {
    FILE_NUM=$((FILE_NUM + 1))
    OUTPUT_FILE="${OUTPUT_BASE}_${FILE_NUM}.md"
    CURRENT_LINE_COUNT=$(write_header "$OUTPUT_FILE" "$FILE_NUM")
}

# Initialize first file
CURRENT_LINE_COUNT=$(write_header "$OUTPUT_FILE" 1)

# Build find command with conditional exclusions
FIND_CMD="find . -type f \( -name \"*.rs\" -o -name \"*.h\" -o -name \"*.toml\" -o -name \"*.py\" \) ! -path \"*/target*/*\""
if [[ "$INCLUDE_EXAMPLES" == false ]]; then
    FIND_CMD="$FIND_CMD ! -path \"*/examples/*\""
fi
if [[ "$SKIP_TESTS" == true ]]; then
    FIND_CMD="$FIND_CMD ! -path \"*/tests/*\""
fi
# Add any custom ignored directories
for dir in "${IGNORE_DIRS[@]}"; do
    FIND_CMD="$FIND_CMD ! -path \"*/${dir}/*\""
done

# Find all .rs, .h, .toml, and .py files, excluding target directory and optionally examples/tests
eval $FIND_CMD | sort | while read -r file; do
    # Get relative path (remove leading ./)
    rel_path="${file#./}"
    
    # Determine language for syntax highlighting
    if [[ "$file" == *.rs ]]; then
        lang="rust"
    elif [[ "$file" == *.toml ]]; then
        lang="toml"
    elif [[ "$file" == *.py ]]; then
        lang="python"
    else
        lang="c"
    fi
    
    # Prepare content in a temporary file
    CONTENT_FILE=$(mktemp)
    
    # Add header with file path
    {
        echo "---"
        echo ""
        echo "## \`$rel_path\`"
        echo ""
        echo "\`\`\`$lang"
    } > "$CONTENT_FILE"
    
    # Add file content, optionally filtering comments and tests
    if [[ "$SKIP_COMMENTS" == true ]] || [[ "$SKIP_TESTS" == true ]]; then
        # Create a temporary file for processing
        TEMP_FILE=$(mktemp)
        cat "$file" > "$TEMP_FILE"
        
        # Filter comments if requested
        if [[ "$SKIP_COMMENTS" == true ]]; then
            if [[ "$lang" == "toml" ]] || [[ "$lang" == "python" ]]; then
                # TOML and Python comments start with #
                grep -v '^\s*#' "$TEMP_FILE" > "${TEMP_FILE}.tmp" && mv "${TEMP_FILE}.tmp" "$TEMP_FILE"
            else
                # Rust and C comments: // and /* */ multi-line blocks
                # First remove multi-line /* */ comments
                sed '/\/\*/,/\*\//d' "$TEMP_FILE" > "${TEMP_FILE}.tmp" && mv "${TEMP_FILE}.tmp" "$TEMP_FILE"
                # Then remove single-line // comments
                grep -v '^\s*//' "$TEMP_FILE" > "${TEMP_FILE}.tmp" && mv "${TEMP_FILE}.tmp" "$TEMP_FILE"
            fi
        fi
        
        # Filter tests if requested (only for Rust files)
        if [[ "$SKIP_TESTS" == true ]] && [[ "$lang" == "rust" ]]; then
            # Use awk to filter out everything from test blocks to end of file
            awk '
                BEGIN { in_test = 0 }
                
                # Once we hit a test marker, ignore rest of file
                /^\s*#\[test\]/ || /^\s*#\[cfg\(test\)\]/ {
                    in_test = 1
                    next
                }
                
                # Only print if not in test section
                !in_test { print }
            ' "$TEMP_FILE" > "${TEMP_FILE}.tmp" && mv "${TEMP_FILE}.tmp" "$TEMP_FILE"
        fi
        
        cat "$TEMP_FILE" >> "$CONTENT_FILE"
        rm -f "$TEMP_FILE"
    else
        cat "$file" >> "$CONTENT_FILE"
    fi
    
    {
        echo ""
        echo "\`\`\`"
        echo ""
    } >> "$CONTENT_FILE"
    
    # Count lines in the content
    CONTENT_LINES=$(wc -l < "$CONTENT_FILE")
    
    # Check if adding this content would exceed the max lines
    if [[ $((CURRENT_LINE_COUNT + CONTENT_LINES)) -gt $MAX_LINES ]] && [[ $CURRENT_LINE_COUNT -gt 0 ]]; then
        start_new_file
    fi
    
    # Append content to current output file
    cat "$CONTENT_FILE" >> "$OUTPUT_FILE"
    CURRENT_LINE_COUNT=$((CURRENT_LINE_COUNT + CONTENT_LINES))
    
    rm -f "$CONTENT_FILE"
done

if [[ $FILE_NUM -gt 1 ]]; then
    echo "✓ Source code compiled to ${FILE_NUM} files: ${OUTPUT_BASE}.md, ${OUTPUT_BASE}_2.md, ..., ${OUTPUT_BASE}_${FILE_NUM}.md"
else
    echo "✓ Source code compiled to $OUTPUT_FILE"
fi
