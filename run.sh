echo "Linear Regression"
echo "python3 numpy_version/main.py --model least --epochs 20 --batch 32 --lr 0.01"

/usr/local/bin/python3.10 numpy_version/main.py --model least --epochs 20 --batch 32 --lr 0.01

echo ""
echo "DNN"
echo "python3 numpy_version/main.py --model dnn --epochs 20 --batch 32 --lr 0.01"

/usr/local/bin/python3.10 numpy_version/main.py --model dnn --epochs 20 --batch 32 --lr 0.01

echo ""
echo "Logistic Regression"
echo "python3 numpy_version/main.py --model logistic --epochs 20 --batch 32 --lr 0.01"

/usr/local/bin/python3.10 numpy_version/main.py --model logistic --epochs 20 --batch 32 --lr 0.01

# echo "Optimizer runs"
# echo "Optimizer - SGD"
# /usr/local/bin/python3.10 numpy_version/main.py --model dnn --optimizer sgd --epochs 20 --batch 32 --lr 0.01

