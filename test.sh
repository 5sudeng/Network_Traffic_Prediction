test_paths=("")

for test_path in "${test_paths[@]}"; do
    python test.py --resume "$test_path" 
done
