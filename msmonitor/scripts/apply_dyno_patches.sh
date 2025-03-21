#! /bin/bash
set -e

apply_ascend_patches() {
    cd ./third_party/dynolog || return 1

    if [ ! -d "../../patches" ]; then
        echo "ERROR: patches directory not found"
        cd ../..
        return 1
    fi

    for patch_file in ../../patches/*.patch; do
        if [ -f "$patch_file" ]; then
            echo "Applying patch: $patch_file"
            git apply --check -p1 "$patch_file"
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to apply patch: $(basename $patch_file)"
                cd ../..
                return 1
            fi
            git apply -p1 "$patch_file"
            if [ $? -ne 0 ]; then
                echo "ERROR: Failed to apply patch: $(basename $patch_file)"
                cd ../..
                return 1
            fi
        fi
    done

    cd ../..
    echo "Successfully applied all Ascend patches"
    return 0
}

apply_ascend_patches