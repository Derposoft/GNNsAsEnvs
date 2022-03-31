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
    python3 test.py --init_health=20 > ./test_logs/"$c"_testat20hp.txt
    python3 test.py --init_health=5 > ./test_logs/"$c"_testat5hp.txt
    python3 test.py --init_health=2 > ./test_logs/"$c"_testat2hp.txt
    cd _checkpoints
    mv ../checkpoints/"$c" ./
done
cd ..
rmdir checkpoints
mv _checkpoints checkpoints
