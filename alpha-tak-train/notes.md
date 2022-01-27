# train configuration results

## 8-128: 2 epoch, lr [0.01, 0.003] with weights +500:
accuracy: 0.07736117702895111
top5 accuracy: 0.18766018035121027
looks really bad, underfit flat placements and didn't learn anything

## 8-128: 2 epoch, lr [0.01, 0.003] no weights:
accuracy: 0.12577123872804935
top5 accuracy: 0.3149501661129568
this is a better score. There seems to be just not enough data to even nearly
fit the model.

# general thoughts

- the model seems to really struggle with early, close to empty boards.
Does it have a problem when it can't extract features from an empty position?
