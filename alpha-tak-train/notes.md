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

## fixed bug!!
I found the mistake, it was indeed in the dataset. The mirrored
symmetries where calculated wrong, so half of the data was scrambled. That is
why it had only 50% the accuracy of the no symmetry model.

## 5-48: 10 epoch, lr 0.01 no-opening no symmetries:
accuracy: 0.11771243811255937
top5 accuracy: 0.33676871779327067

## 5-48: 10 epoch, lr 0.01 no-opening all symmetries:
accuracy: 0.15202759549049302
top5 accuracy: 0.4230607437321218
Surpasses no symmetry in first epoch, acc. after 1 epoch:
accuracy: 0.1293117953895339
top5 accuracy: 0.36739301138594427
No sign of overfitting.
This is definitely enough data to drop LR during training, but to get the
direct comparison between both dataset options I don't change LR here.

I put bias back in, as Lc0 also uses bias on convolutions.

## 8-128: 6 epoch, lr 0.01 (drop after every 3):
accuracy: 0.30242592592592593
top5 accuracy: 0.6657592592592593
This is insane. The model was too small to fit everything before. Adding bias
and filters + extra layers brought the net on another level. This might
even be able to play a game with just policy head?!?

## 8-128 3 epoch, all data(human + bot):
tested on human dataset:
accuracy: 0.32136260301221936
top5 accuracy: 0.6934143222506394

tested on bot dataset:
accuracy: 0.35429017160686427
top5 accuracy: 0.7008060322412897

Because larger net performed so much better, I am gonna train an even bigger net.
Benchmarks on unknown google colab GPU:
17s for 99k positions (batchsize 128-512),
18s for 99k positions (batchsize 64),
24s for 99k positions (batchsize 32)

## 16-192 3 epoch, all data (human + bot):
tested on human dataset:
accuracy: 0.341361182154021
top5 accuracy: 0.7200554134697357

Was still improving after 3 epochs. I can try again with more epochs. But this
already takes more then 1h on Tesla T4 (equivalent of RTX 2070), so I might
return to smaller sized net. 8-128 or 10-128, with more epochs.
This large net needs more data to be worth the extra computations. Optimally, I
have like 25k games (equivalent of 200k games because of symmetries) to match
small Bad Gyal training dataset.

## 10-128 trained on 21k tiltak selfplay games:
acc:  0.38003332407664536
top5 acc:  0.7795056928630936
strongest net yet.

The bot has a really strong opening, but is much worse later on. Accuracy drops
down to only 0.2 even. I will try to improve on that by balancing dataset,
supplying equally early game positions instead of twice as many compared to
middle/late game positions

## 10-128 21k tiltak (3 epoch), balanced dataset:
acc:  0.42035469710272166
top5_acc:  0.8334047410008779
This definitely helped with late game moves. Also, a huge improvement was during
3rd epoch, so more data/more epochs might still substantially improve accuracy.
I will try that with a SE-net and hopefully get mind-blown again.

# general thoughts
- the model seems to really struggle with early, close to empty boards.
Does it have a problem when it can't extract features from an empty position?
-> Resolved after Bias!
- I am not sure about the fully convolutional policy head.
-> I have yet to run a test to compare, but as results are very promising for
convolutional policy head I haven't yet.
