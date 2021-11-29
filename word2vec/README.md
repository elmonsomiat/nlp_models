# Word2Vec

Example to train
```python
 python train.py --text  "Here kids will write everyone's wonderful long text"
```
to use a file:
```python
python train.py --epochs 20 --file ../data/to_kill_a_mocking_bird.txt
```
For a prediction
```python
 python predict_from_context.py --text  "Here kids will write everyone's long text" --train 0
```

Find similar words
```
python similar_search.py --text "shoes" 
```