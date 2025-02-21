BASE_PATH=$(realpath ../examples/bank_conflict/matrix_transpose)

./maestro --name matrix_transpose \
      --script "$BASE_PATH/build.sh" \
      -vv \
      -- "$BASE_PATH/matrix_transpose_inst"