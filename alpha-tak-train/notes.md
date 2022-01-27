# train configuration results
training on 1596 human vs. human games from playtak, all games with both players
rated above 1600.

## 8-128: 2 epoch, lr [0.01, 0.003] with weights +500:
accuracy: 0.07736117702895111
top5 accuracy: 0.18766018035121027
looks really bad, underfit flat placements and didn't learn anything

## 8-128: 2 epoch, lr [0.01, 0.003] no weights:
accuracy: 0.12577123872804935
top5 accuracy: 0.3149501661129568
this is a better score. There seems to be just not enough data to even nearly
fit this model size.

## 5-48: 10 epoch lr 0.01:
accuracy: 0.13317513051732321
top5 accuracy: 0.3502610346464167
I think this overfit. Still, this shows that
- dataset is too small
- small models definitely work no problem

## 5-48 2 epoch, lr [0.01, 0.001] (all symmetries):
accuracy: 0.07974703069566559
top5 accuracy: 0.28012909502960404
This looks like something went wrong with creating symmetries. this is 8x data,
should get better result.
LR too high? train again with lower starting LR. If that
doesn't help, dataset is bad. (bad gyal can fit human chess games, same model)
switch to/include bot matches?

## 5-48: 2 epoch, lr [0.003, 0.0003] (all symmetries):
Doesn't improve, cancelled run. Dataset looks fine, though. Lower accuracy seems
to be because players have orientation preference. Removing first 4 plies from
dataset for this and other reasons. Training for 10 epochs with and without
symmetries from 5th ply onward and comparing results.

## 5-48: 10 epoch, lr 0.01 no-opening no symmetries:
accuracy: 0.11771243811255937
top5 accuracy: 0.33676871779327067

## 5-48: 10 epoch, lr 0.01 no-opening all symmetries:


# general thoughts
- the model seems to really struggle with early, close to empty boards.
Does it have a problem when it can't extract features from an empty position?
- I am not sure about the fully convolutional policy head.
