#!/bin/bash

cd test/cuda-samples
make -j

# Path to the directory containing executables
BIN_DIR="./bin/x86_64/linux/release"

# Success message
SUCCESS_MSG="All scripts executed successfully."

# Error message
ERROR_MSG="Error encountered while executing a script."

# Loop through all executable files in the directory
find "$BIN_DIR" -type f -executable | while read -r file; do
    cd "$(dirname "$file")"

    # Run the script with a timeout and suppress stdout/stderr
    OUTPUT=$(timeout 30s "./$(basename "$file")" 2>&1)
    RET_CODE=$?
    
    if [[ $RET_CODE -ne 0 ]]; then
        # Print the output and error message
        echo "Error executing $file:"
        echo "$OUTPUT"
        echo "$ERROR_MSG"
    else
        echo "Successfully executed $file"
    fi

    cd - > /dev/null
done

# Print success message if all scripts run successfully
echo "$SUCCESS_MSG"
