# Red Sox Problem Set

# Getting Started
This code was created and run on Linux (Ubuntu 20.04), I do not guarantee that it will run on other operating systems. Additionally, because of the somewhat large size of the data, and because I have a local GPU, I use the RAPIDSAI suite of packages for easy GPU computing. Therefore, this code requires a NVIDIA GPU with CUDA enabled to run. See [https://rapids.ai/start.html](https://rapids.ai/start.html) for installation via conda.

# Model
I decided to use a tree based model for this classification problem. The main reason for this was the project requirement to report "what inputs to your model(s) seem to be driving that prediction for that
particular pitch the most?" With tree based methods, it is very easy to see which features are most important in the classification. It is also simple to implement and relatively fast to train. Unfortunately, after spending some time tuning this model, it still does not give great results. If I were to continue to work on this project the first thing I would do is try a different type of model.

# Results
Because the data is imbalanced, I had to decide how to balance overall accuracy and accuracy per class for my final model. I decided to choose a model with a high True Positive rate for predicting a swing and miss; at the cost of a high false positive rate for predicting not a swing and miss. I mainly made this decision because the problem set is interested in determining what factors lead to swing and misses. A model that is biased more towards predicting swing and misses will be better suited for this.

The final model chosen had a validation accuracy of just 67%. This is very low, but is because I heavily weighted the swing and miss results. The most naive model of just predicting that every pitch does not result in a swing and miss results in a validation accuracy of about 88%. While this is high, it is a obviously a useless model that gives no insight; the accuracy rate for pitches that are swings and misses is 0%. My model gives a validation accuracy of 71% given that the pitch is a swing and a miss. I believe that both of those numbers can be improved with more time.

The prediction results for the holdout set can be seen in Predictions.csv.

The top three pitcher-pitch combinations that are likeliest to produce a swing and miss when thrown by the pitcher are

| Pitcher | Pitch Type |
|------|------|
| 354630 | SL |
| 305270 | CB |
| 348223 | CH |

The effectiveness of all of these pitches is at least partially driven by the most weighted inputs to the model, which are in order:

| Input |
|------|
| PlateLocX |
| PlateLocZ |
| ReleaseVelocityX |
| PitchBreakVert |
| PitchBreakHorz |
| ReleaseLocX |
| ReleaseVelocityZ |
| ReleaseLocZ |
| SpinRate |
| ReleaseLocY |

This is based purely on percentages and two of the pitchers, 354630 and 305270 only threw 50 and 75 of those pitch types respectively. For more details see the results section of main.ipynb.

