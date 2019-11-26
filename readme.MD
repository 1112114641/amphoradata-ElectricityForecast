<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
# Predicting Electricity Spot Prices using [Amphora Data](https://amphoradata.com) and Koz.ai
=======
# Predicting Electricity Spot Prices using [Amphora Data](https://amphoradata.com) and [Koz.ai](https://www.koz.ai)
>>>>>>> Amphora Data section (#3)
=======
# Predicting Electricity Spot Prices using [Amphora Data](https://amphoradata.com) and [Koz.ai](https://koz.ai)
To skip the initial explanations, and dive into the project follow the [quick start manual](#quick-start-the-project).
>>>>>>> Markus edits (#4)
=======
# Predicting Electricity Spot Prices using [Amphora Data](https://amphoradata.com) and [Kozai](https://koz.ai)
<<<<<<< HEAD
<<<<<<< HEAD
 > To skip the initial explanations, and dive into the project follow the [quick start manual](#quick-start-the-project).
>>>>>>> updated ordering of readme.md
=======
 > To skip the initial explanations and project descriptions and dive straight into the project follow the [quick start manual](#quick-start-the-project).
>>>>>>> updated ipynb-file:
=======
 > To skip the initial explanations and project descriptions and dive straight into the project follow the [quick start manual](#quick-start).
>>>>>>> Fix internal link (#6)

In South Australia, electricity is predominantly sourced from renewable energy sources such as sun and wind. As both wind speeds and sun insolation vary greatly, so do electricity prices. In an effort to model the electricity spot price spikes in South Australia, data from the [Amphora Data](https://amphoradata.com) agricultural data repository was ingested into the [Kozai](https://koz.ai) data science platform for modelling and predictions, resulting in a good predictor for price spikes based on a convolutional neural net.

## Who we are

### Better data, and more of it

Accurate predictions, especially in forecasting problems, require large amounts of data. Making a prediction on something not seen before by the model, will be difficult in most situations. Often, more data is not sufficient though, and other data, other variables help to improve model performance.

**Amphora Data** is a platform to discover, package, and trade data relevant to agriculture, either in real-time, or historical. An Amphora is an abstraction on data that allows us to generalise on its contents. Each Amphora generally contains a small but useful dataset that is consistently being updated. Amphorae provide a standard interface to describing, accessing, and monetising data so that each additional data source is easier to include than the last.

The screenshot below shows the Amphora web application displaying metadata from a Meat and Livestock Australia data feed. In fact, any organisation on Amphora Data can create and trade arbitrary data as they like, and the number of Amphorae is growing consistently. All the data used in this project are available at [beta.amphoradata.com](https://beta.amphoradata.com)

![Labelled screenshot of Amphora Data web application](images/screenshot-amphoraBreakdown.png)
**Figure 1:** Screenshot of the Amphora Data web app.

Amphora Data also provides Python (as used in this project), and Dotnet SDKs, and an API specification, so you can integrate data into your own applications or data science tools and processes.

Amphora Data is currently in private beta. For access visit [amphoradata.com/contact-us](https://amphoradata.com/contact-us)

###  Simplify Data Science with [Kozai](https://koz.ai)
**Kozai** is a serverless self-service cloud environment for easy, simplified data science development and deployment.
With more data and larger projects, local compute power will soon become limiting and correspondingly waiting times will increase. [Kozai](https://koz.ai) offers the solution to computers overheating and fans reaching the noise level of a vacuum cleaner! It allows you to scale your projects easily from a single CPU, or GPU, to an entire cluster for plentiful compute power.
Also, the built-in git support makes version control simple and allows for easy collaboration on a project.
Thus, Kozai makes moving between different work locations easy, removes issues with fickle updates of environments and packages, as the Kozai environment is identical, irrespective of where you log in from, or which of your users log in.

Kozai offers the perfect platform for projects like the one described here, while also being reasonably priced and even offering a pay-as-you-go model.

![Screenshot of current state of Kozai UI](images/KozaiUI.png)
**Figure 2:** The Kozai UI, with labels showing some key intrinsic features.

Kozai is currently in the private beta phase, going into public beta in Q2 2020. To register to the mailing list for information and updates visit [https://koz.ai](https://koz.ai).

## Electricity Spot Prices in South Australia

Predicting the price of electricity is increasingly important. The wholesale market of electricity is extremely variable and many companies and organisations are highly exposed. This is an issue in agribusiness, where many farmers are facing decreasing profit margins and price spikes can cause great difficulties.
While farmers can benefit from periods of prices as low as $50 /MWh or even less, they also face prices of up to $14000 /MWh. There is a fundamental conflict - businesses want price certainty to plan their daily operations, but the Australian National Electricity Market (NEM) requires price variability to shift production along the supply/demand curve. Here, we present a small contribution to resolving this conflict with methods and tools for better predicting price fluctuations, which can help companies manage the use of their electricity accordingly.

### What is the NEM?

The Australian National Electricity Market (NEM) is a wholesale electricity market in which generators sell electricity and retailers buy it to sell on to consumers. There are over 100 generators and retailers participating in the market, so it’s highly competitive and therefore an efficient way of maintaining relatively competitive electricity prices in the wholesale market.

All electricity sales are traded through the NEM. It is a wholesale market and prices fluctuate in response to supply and demand at any point in time. The NEM is split into 5 regions (QLD, NSW, VIC, TAS, SA), and prices are set per-region.

![Map of the NEM regions](images/map-NEM.png)
**Figure 3:** Map of the NEM regions: South Australia (grey), Victoria (green), Tasmania (red), New South Wales (blue), and Queensland (yellow).

### What do the prices look like?

> All plots and data below are available at [Amphora Data](https://amphoradata.com).

Electricity prices are set every 5 minutes (the ‘dispatch price’) and 6 dispatch prices are averaged every half hour to generate the ‘spot price’. Below are the prices for an arbitrary week in South Australia. We observe that the price remains relatively close to $50, although occasionally dropping sharply and [becoming negative](https://www.afr.com/companies/energy/why-electricity-spot-prices-are-hitting-zero-20190723-p52a08). At these times, consumers of electricity are actually paid to consume.

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
Best case scenario, you need an automated and robust prediction of the price, and an automated go/no-go decision framework. [Amphora Data](https://AmphoraData.com) and Koz.ai have partnered to develop this open source experimentation tool for achieving just that. We hope you find it useful :)
=======
# Predicting Electricity Spot Prices using Amphora Data
Give outline of what happened

Talk about why amphoradata, why spot prices of interest
>>>>>>> updates to EForecast.yml, readme.MD for setup of env
=======
Best case scenario, you need an automated and robust prediction of the price, and an automated go/no-go decision framework. [Amphora Data](https://amphoradata.com) and Koz.ai have partnered to develop this open source experimentation tool for achieving just that. We hope you find it useful :)
>>>>>>> Amphora Data section (#3)

## quick start on the project
1. Run `git clone https://github.com/1112114641/amphoradata-ElectricityForecast.git` in kozai terminal, or download from github `https://github.com/1112114641/amphoradata-ElectricityForecast` and then manually upload to kozai
2. Run `cd 0_setup`, then execute `export usrname='yourAmphoraUsername'`, and `export password='yourAmphoraPassword'` to set the amphora data password and username
3. Run `conda env create -f EForecast.yml` to setup the environment, install python packages and then activate the env with `conda activate EForecast`
 - if a `permission denied` error occurs, change rights with `chmod a+x EForecast.yml` to make the file executable, run 3., then change back to base dir with `cd ..`
4. Take care to ensure the variables $usrname and $password are set, so connection to amphora API possible. To test this, run `echo $usrname` or `echo $password` in your shell
5. Open the `ipynb` file in the base directory, which is now ready to be run

<<<<<<< HEAD
## talk about data importance, why amphora
<<<<<<< HEAD
<<<<<<< HEAD
For data science, and specifically for forecasting problems, accurate predictions require large amounts of data. Afterall, making a prediction on something not seen before by the model, will be difficult in most situations. Amphora offers...
=======
For data science, and specifically for forecasting problems, accurate predictions require large amounts of data. Making a prediction on something not seen before by the model, will be difficult in most situations. Often, more data is not sufficient though, and other data, other variables help to improve model performance. Amphora offers...

![Example of the layout of the contents of an Amphora](4_images/amphoradataUI.png)
>>>>>>> updates to EForecast.yml, readme.MD for setup of env
<!-- screenshot of UI of amphora + description-->
=======
For data science, and specifically for forecasting problems, accurate predictions require large amounts of data. Afterall, making a prediction on something not seen before by the model, will be difficult in most situations.
<!-- screenshot of UI of amphora-->
>>>>>>> update on readme, changes to tools.py
=======
# Better data, and more of it.
=======
Best case scenario, you need an automated and robust prediction of the price, and an automated go/no-go decision framework. [Amphora Data](https://AmphoraData.com) and [Koz.ai](https://koz.ai) have partnered to develop this open source experimentation tool for achieving just that. We hope you find it useful :)
=======
Best case scenario, you need an automated and robust prediction of the price, and an automated go/no-go decision framework. [Amphora Data](https://amphoradata.com) and Koz.ai have partnered to develop this open source experimentation tool for achieving just that. We hope you find it useful :)

# Who we are
>>>>>>> More Contributions (#5)

## Better data, and more of it.
>>>>>>> Markus edits (#4)
=======
![Plot of normal electricity price and scheduled generation in a week](images/plot-normalElectricityPriceAndGen-SA.png)
**Figure 4:** Normal electricity price and scheduled generation in a week for SA.

Now compare the 'normal' week above, to the week that immediately followed it in December of 2019. Clearly things are very different. The 'normal' variability ~$50 is no longer visible, as the graph is dominated by a huge spike on December 19, when the price jumped to over $8,000, approximately two orders of magnitude larger than the maximum for the days either side. We also observe that the maximum scheduled generation has doubled compared with the previous week.
![Plot of abnormal electricity price and scheduled generation in a week](images/plot-spikeElectricityPriceAndGen-SA.png)
**Figure 5:** Abnormal electricity price and scheduled generation in a week for SA.

So what's going on? A major driver of electricity demand (and therefore supply) is the weather. The plot of temperature below shows that Adelaide was experiencing a serious heatwave from December 17 to December 20, with the price spike occurring the evening before the hottest and last day. Keen observers will also note that the minimum overnight temperature was high at 28 degrees.
>>>>>>> updated ordering of readme.md

We've highlighted the same period for each day of the heatwave (7 - 9 pm), which coincides with the price spike on the last day. Temperatures are significant, because higher temperatures mean more air-conditioning (i.e. greater consumption) and often less efficient power plants. Wind speed is also significant because of the high use of wind energy in South Australia.

![Plot of temperature and wind speed during abnormal week](images/plot-weatherAroundSpike-Adelaide.png)
**Figure 6:** Temperature and wind speed during abnormal week for SA.

### Counting the cost

If your business was regularly running a plant at 7pm, and that plant consumes ~250 kW of power, on a normal day you'd spend about $15/hour to run the plant. Compare that to the abnormal day on December 19. On that day, at 7pm, you'd spend $1,500 in a single hour!

### Why hasn't this happened to me?

Luckily for regular consumers, electricity retailers (think AGL, Origin etc.) protect you from price variability by hedging against shortfalls in generation. Therefore, no ridiculously expensive days. Consumers *are* paying for the price spikes, it's just averaged out over the entire year.

<<<<<<< HEAD
<<<<<<< HEAD
## Quick start the project

<<<<<<< HEAD
>>>>>>> Amphora Data section (#3)
=======
To get started with the project, follow these steps to get set up:
>>>>>>> Markus edits (#4)

1. Run `git clone https://github.com/1112114641/amphoradata-ElectricityForecast.git` in kozai terminal, or download from github `https://github.com/1112114641/amphoradata-ElectricityForecast` and then manually upload to kozai
2. Run `cd 0_setup`, then execute `export usrname='yourAmphoraUsername'`, and `export password='yourAmphoraPassword'` to set the amphora data password and username in your terminal`
3. Run `conda env create -f EForecast.yml` to setup the environment, install python packages and then activate the env with `conda activate EForecast`
 - if a `permission denied` error occurs, change rights with `chmod a+x EForecast.yml` to make the file executable, run 3., then change back to base dir with `cd ..`
4. Take care to ensure the variables $usrname and $password are set, so connection to amphora API possible. To test this, run `echo $usrname` or `echo $password` in your shell
5. Open the `ipynb` file in the base directory, which is now ready to be run

<<<<<<< HEAD
<<<<<<< HEAD
 - comment on management of teams? (covered during meeting today)
 - management of instances?
 - compute CPUs/GPUs available, multi processor
 - running of task montoring idk
=======
>>>>>>> Markus edits (#4)

###  Simplify ([Koz.ai](https://koz.ai)) your life
=======
##  Simplify ([Koz.ai](https://koz.ai)) your life
>>>>>>> More Contributions (#5)
=======
### So why would anyone want to be exposed to the variable price?
>>>>>>> updated ordering of readme.md

Remember that during the normal week of operations, the price actually went negative, meaning that anyone exposed and consuming electricity during that period was actually paid by the market. For our imaginary business, for the hour when price dropped to $-30, they would have actually been paid about $7 to run their plant - a $22 improvement on the typical situation.

You may also wish to be exposed if you have atypical usage patterns, because as long as you don't consume during periods of high prices, your total electricity bill will be significantly smaller because you aren't paying a retailer to hedge the market for you.

### How to manage costs when exposed to the wholesale price

Best case scenario, you need an automated and robust prediction of the price, and an automated go/no-go decision framework. [Amphora Data](https://amphoradata.com) and [Kozai](https://koz.ai) have partnered to develop an open source tool for achieving just that. We hope you find it useful :)

# Code and Data

## Quick start

To get started with the project, follow these steps to get set up:

1. Run `git clone https://github.com/1112114641/amphoradata-ElectricityForecast.git` in kozai terminal, or download from github `https://github.com/1112114641/amphoradata-ElectricityForecast` and then manually upload to kozai.
2. Run `cd 0_setup`, then execute `export usrname='[YourAmphoraUsernameHere]'`, and `export password='[YourAmphoraPasswordHere]'` to set the amphora data password and username.
3. Run `conda env create -f EForecast.yml` to setup the environment, install python packages and then activate the environment with `conda activate EForecast`.
4. Check that the variables $usrname and $password are set, so that the connection to amphora API is possible. To test this, run `echo $usrname` or `echo $password` in your terminal shell.
5. Open `Amphora-ElectricityPrice_Production.ipynb` in the base directory, which is now ready to be run

### Wrapper around Amphora Data Python SDK

A wrapper [APIfetch.py](./src/APIfetch.py) for the Amphora Data API was created to help the eventual automation of the data download, forecasting and data upload.  The wrapper facilitates these three steps for users, only requiring them to state the desired dates and amphora, while taking care of everything else in the process.

## Into the Data Science

The project was started with some exploratory analysis of the downloaded spot prices and weather data from Amphora Data. As a proof-of-concept / baseline, a multiple linear regression model was then built to predict spot prices for electricity in South Australia (SA). These simple models often show a good performance in a wide range of situations and offer an indication of the importance of new features as they are added to the model. Then, more complex and involved models were developed and tested, measured against this baseline.

### Exploratory Data Analysis (EDA)

As a start, as in most of my projects, data was checked for completeness, and if there had been any missing values, they would have been interpolated with splines. To capture the temporal nature of the data, new features were created consisting of the discrete t<sub>n</sub>-t<sub>n-1</sub> time differences.

The correlations between the SA spot prices and all other variables were tested, yielding the result that the SA spot prices behave the most similar to the `scheduledGeneration` in Victoria (VIC) and SA. The next most correlated features were weather-related, namely those that influence solar cell and/or wind farm performance.

```
>>>corr_SA.price_SA.sort_values(ascending=False).head(10)
price_SA                   1.000000
scheduledGeneration_VIC    0.388196
scheduledGeneration_SA     0.369061
d_price_SA                 0.331827
rainProb_SA                0.225282
d_rainProb_SA              0.224557
d_windDirection_SA         0.215873
cloudCover_SA              0.207847
windDirection_SA           0.206184
d_cloudCover_SA            0.205422
Name: price_SA, dtype: float64
```
Data from the other states was included for the SA price modelling effort, as their electricity grids are connected, thus influencing each other.

Similarly, the prices in NSW and QLD follow a similar pattern of mutual dependence. In an analysis as for SA above, the dominant features are the respective `scheduledGeneration` for NSW and QLD, as expected in connected electricity markets. The small influence between SA/VIC on the one side, and QLD/NSW on the other, lies in their geographic separation, and different electricity generation sources.

Adjusting the linear model to contain only these top nine features already improved the performance, showing a about 30% lower mean square error (mse):
```
>>>print('RMSE with correlation features:', rmse_corr, '$/MWh\n')
RMSE with correlation features: 207.58 $/MWh
>>>print('RMSE w/o correlation features: ', rmse_plain, '$/MWh')
RMSE w/o correlation features:  281.23 $/MWh
```
For both the random forest regressor, as well as the deep learning models tested for model selection, a similar improvement in performance manifested using only these highly correlated features, or a similar selection of a subset of features.

#### Distribution of data

Especially for classical machine learning methods the distribution of data matters, where normal distributed data leads to faster algorithm convergence and better predictions. In this case, fortunately, most data was fairly close to either a shifted normal, or log-normal distribution as can be seen in a pairplot. Correspondingly, no data transformation was undertaken, although it is a possible improvement factor to consider for future modelling efforts.

### Feature engineering

The amount of data at the start of the project was limited in terms of typical forecasting applications, with less than 30 days worth of data available. As there was great variation in the performance of new features as fresh data came in daily, the creation of such features to help improve the modelling was not actively pursued to the later stages.

<<<<<<< HEAD
### prelims: amount of data
=======
 - compute CPUs/GPUs available, multi processor
 - running of task montoring idk

<!-- screenshot of prelim UI kozai-->

### talking about the datascience
structure of eda, API access to amphora, revisit modelling/multiple models, choice of model by lowest error criterion

#### prelims: amount of data
>>>>>>> update on readme, changes to tools.py
data was limited so results jumpy, ideally at least 3yrs of data, not just 3 months (possible project on the murray river basin water levels)
=======
### Model selection
>>>>>>> Markus edits (#4)

The central aim for model selection was (1) prediction of spikes in spot prices for SA, and (2) describing _all_ SA spot prices as accurately as possible. Given the goal of this project, (1) took precedence over (2).

<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
<<<<<<< HEAD
### prelims: model selection
Often, the central question for data science is the choice of the model, where several competing factors have to be considered: between explainability, model performance, model inference time, and model train time.
linear model, RFR, Dense, CNN model, all compared for the chose criterion of rmse, as large errors are punished more severly
=======
The best performing model with the current amount of data (~45days) was the random forest of which the predictions are shown [below](#predictions), as selected by the overal minimal mse.
>>>>>>> Markus edits (#4)
=======
The best performing model with the current amount of data (~45days) was a small convolutional neural net (CNN) optimising for each state individually, as selected by the overal minimal mean mse over 24 timesteps. The close second most accurate model being the random forest.

As this model will be continuously run once a day, the model selection step will be repeated after every run, to ensure best model performance.
>>>>>>> updates, bug fixes, etc
=======
The best performing model with the current amount of data (~45days) was a convolutional neural net (CNN) optimising by SA spot prices, as indicated by the overal minimal mean mse over the 24 prediction timesteps. The close second most accurate model was a very similar model, optimising for each state concurrently.

The model performance was confirmed by comparing the mean mse over the 24 predicted values for all models.
>>>>>>> final commit
=======
The best performing model with the current amount of data (~45days) was a convolutional neural net (CNN) optimising by SA spot prices, as indicated by the overal minimal mean mse over the 24 prediction timesteps. The second most accurate model with very similar performance optimised for each state concurrently.

The best model performance for final predictions was confirmed by comparing the mean mse over the 24 predicted values for all models and architectures.
>>>>>>> 'minor cosmetic changes'
=======
The best performing model with the current amount of data (~45days) was a convolutional neural net (CNN_4) optimising by SA spot prices, as indicated by the overall minimal mse over the 24 prediction timesteps. This model yielded a standard error of 69 $/MWh for the SA spot price predictions. The second most accurate model with very similar performance optimised for each state concurrently (CNN_2).
=======
Often, the central question for data science is the choice of the model, where several competing factors have to be considered: explainability, model performance, model inference time, and model train time.
<<<<<<< HEAD
The latter two factors were not a concern for this specific project thanks to Kozai! Hence a variety of different models and architectures were tested, among them simple linear models (Lin), random forests (RFR), but also dense (NN_3), recurrent (RNN) and convolutional neural networks (CNN_2 and CNN_4). The linear model and RFR were chosen, as they often show good performance, especially so on small datasets, and allow for easy explainability of the models outputs. The deep learning models were chosen, because they either have are explicitly geared towards this kind of problem, like the recurrent neural net and the [causally padded](http://arxiv.org/abs/1609.03499) convolutional neural net, or have well-performing historic use cases for similar problems.
>>>>>>> updated ipynb-file:
=======
The latter two factors were not a concern for this specific project thanks to Kozai! Hence a variety of different models and architectures were tested, among them simple linear models (Lin), random forests (RFR), but also dense (NN_3), recurrent (RNN) and convolutional neural networks (CNN_2 and CNN_4). The linear model and RFR were chosen, as they often show good performance, especially so on small datasets, and allow for easy explainability of the models outputs. The deep learning models were chosen, because they either are explicitly geared towards this kind of problem, like the recurrent neural net and the [causally padded](http://arxiv.org/abs/1609.03499) convolutional neural net, or have well-performing historic use cases for similar problems.
>>>>>>> updated typos

#### Model training optimisation

To accelerate model training and obtain better results for the deep learning models, the [Adam](https://arxiv.org/abs/1412.6980) optimiser learning rate was fine tuned, by training the model for one epoch, and stepping through a [nested interval](https://en.wikipedia.org/wiki/Nested_intervals) optimisation, to find the optimal learning rate. For all, except the RNN, the best learning rate was a factor two larger than the standard setting of `1E-3`, leading to slightly faster convergences.

#### Model error
For all these models, the optimisation criterion was the mean square error (mse). This criterion was chosen as large outliers are undesirable and the mse is easily interpretable, if its square root is considered, as an error in units of $/MWh.
Each model was trained on features _x_, with the corresponding spot prices _y_ lagged by 1 to 24h, to be able to predict 1 to 24 time steps into the future, with the results, listed below:
```
>>>print(mse)
Lin: 	43090
RFR:	78459
RNN: 	17158
CNN_2:  6350
CNN_4:  5832
NN_3: 	369214
```
>>>>>>> updated ordering of readme.md

The reason for these 24h prediction windows is rapidly decreasing forecast quality for longer periods, and a more accurate prediction of the windows between 4-9pm, and 2-9am, where most price spikes occur. With shorter prediction windows model accuracy, in general, improves but the usefulness of the model decreases.

The best performing model with the current amount of data (~45days) was a convolutional neural net (CNN_4) optimising by SA spot prices, as indicated by the overall minimal mse over the 24 prediction time steps. This model yielded a standard error of 76 $/MWh for the SA spot price predictions. The second most accurate model with very similar performance optimised for each state concurrently (CNN_2) with a standard error of 79 $/MWh.



## Predictions
![Predictions for the 17th - 23rd Dec 2019](images/Spike.png)
![Predictions for the 25th - 29th Nov 2019](images/trough.png)

**Figure 7:** Predictions for some of the electricity spot price spikes in SA, focusing on two periods with many price negative and positive spot price spikes. The negative spot prices are not mapped well onto the predictions, whereas the positive spikes are well mapped.

Predictions made by this model showed to be good for the prediction of large trends in spot price, like spikes, if they cross a certain threshold. For the precise determination of spot prices, model performance tended to be off between $ 10-30, as visible from the two plots above in Figure 7.

On a Kozai single GPU instance, training the most accurate model was over 5.9x faster than on a 8 core CPU instance, ensuring a fast training and evaluation time.
<<<<<<< HEAD
<!-- 2x4 grid of date (4x) vs ((QLD,NSW, True),((VIC,SA,True)))-->
=======
#### prelims: model selection
Often, the central question for data science is the choice of the model, where several competing factors have to be considered: between explainability, model performance, model inference time, and model train time.
linear model, RFR, Dense, CNN model, all compared for the chose criterion of rmse, as large errors are punished more severly

### predictions
predictions showed to be reasonable for the prediction of trends in spot prices, whereas the precise value mostly was off.
<!-- 2x4 grid of date (4x) vs (true, predict)-->
>>>>>>> update on readme, changes to tools.py
=======
>>>>>>> updated ordering of readme.md

## Final note

In time, given a larger amount of data (~1-2 years worth), feature engineering and more fine tuned models are likely to improve predictions from spot price spike prediction to levels, where the actual price at any point in time can be predicted more accurately.

Focussing on the more regular day-to-day price variations, rebinning from 1h intervals into larger bins, e.g. 2, 4, 6h bins, can bring out further characteristics in the data and make it easier for the model to pick on different characteristics of the evolution and different parts of the signal, further improving model performance.

For a larger amount of data, rare events like spot price spikes will then have to be more carefully modelled for, to ensure the models correct predictions, in order not to drown out these rare events. For this end, training can be specifically focussed on these, or artificial data can be created to "enrich the price spike data".
In combination with both a larger dataset, and enriched data, other neural net architecture that have recently shown great performance in related tasks, like [transformers](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html), could become an interesting option.

