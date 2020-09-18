# Predicting Electricity Spot Prices using [Amphora Data](https://amphoradata.com) and [Kozai](https://koz.ai)
Repo following the [Medium blogpost](https://medium.com/kozai/predicting-electricity-spot-prices-using-amphora-data-and-kozai-431033694d28).
## quick start on the project
1. Run `git clone https://github.com/1112114641/amphoradata-ElectricityForecast.git` in kozai terminal, or download from github `https://github.com/1112114641/amphoradata-ElectricityForecast` and then manually upload to kozai
2. Run `cd 0_setup`, then execute `export usrname='yourAmphoraUsername'`, and `export password='yourAmphoraPassword'` to set the amphora data password and username
3. Run `conda env create -f EForecast.yml` to setup the environment, install python packages and then activate the env with `conda activate EForecast`
 - if a `permission denied` error occurs, change rights with `chmod a+x EForecast.yml` to make the file executable, run 3., then change back to base dir with `cd ..`
4. Take care to ensure the variables $usrname and $password are set, so connection to amphora API possible. To test this, run `echo $usrname` or `echo $password` in your shell
5. Open the `ipynb` file in the base directory, which is now ready to be run