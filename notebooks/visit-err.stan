data {
    int n_stars;
    int n_visits[n_stars];
    int total_n_visits;

    vector[total_n_visits] rv;
    vector[total_n_visits] rv_var;
    vector[total_n_visits] rv_snr;
}

transformed data {
    int cum_n_visits[n_stars];

    cum_n_visits[1] = 0;
    for (n in 2:n_stars) {
        cum_n_visits[n] = n_visits[n-1] + cum_n_visits[n-1];
    }

}

parameters {
    real mean_rv[n_stars];
    real a;
    real<lower=-5, upper=0> b;
    real<lower=-7, upper=0> lns;
}

transformed parameters {
    real s;
    s = exp(lns);
}

model {
    real var_;
    int K;

    lns ~ normal(-2.3, 1)T[-7, 0];

    for (n in 1:n_stars) {
        K = cum_n_visits[n] + 1;
        for (k in K:K + n_visits[n] - 1) {
            var_ = rv_var[k] + s*s + a * pow(rv_snr[k], b);
            target += normal_lpdf(rv[k] | mean_rv[n], sqrt(var_));
        }

        // Prior on mu_n
        target += normal_lpdf(mean_rv[n] | 0, 75);
    }
}
