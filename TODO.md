# TODO

- Check how time is handled for cases with >3 planets
    1. tseries[i, 0] - I think this should be the same for ALL time series (/minP).
    2. For each trio, returns T_inst/min_{all planets}(P). Thus, we don't need to renormalize that - should be fine.

# Done

- (DONE) Test training of model, evaluation of model.
- (DONE) Put up time series data
- (DONE) Put up training code
- (DONE) Set up 5-planet plotting scripts
- (DONE) Unify with spock model
- (DONE) Make globus links
- (DONE) Add FeatureRegressor to spock master.
- (DONE) Get extended tseries to only query 1/10th timesteps, rather than doing it afterwards
