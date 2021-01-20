# predict_api_by_flask-tensorflow

###### Train

```
python3 train.py
```

###### Server

```
python3 predict.py
```





>   **API Request** 

> POST：http://127.0.0.1:5555/predict

> - Body：

  ```
 {
    "g1":1,
    "g2":0,
    "g3":1,
    "g4":0,
    "g5":1,
    "g6":1,
    "g7":0,
    "g8":1,
    "g9":1,
    "g10":1
}
  ```

> - Response：

  ```
  HTTP/1.1 200 OK
  {
    "prediction": "[[0.8754984]]",
    "success": true
  }
  ```

