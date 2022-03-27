mkdir -p checkpoints
mv checkpoints _checkpoints
mkdir checkpoints
mkdir -p test_logs
cd _checkpoints
for c in ./*/; do
    echo "$c"
    c="${c:2:-1}"
    mv "$c" ../checkpoints
    cd ..
    python3 test.py > ./test_logs/"$c".txt
    cd _checkpoints
    mv ../checkpoints/"$c" ./
done
cd ..
rmdir checkpoints
mv _checkpoints checkpoints
