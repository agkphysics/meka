package meka.classifiers.multilabel.meta.boosting.llda;

import java.io.*;


/**
 * Estimation of training model.
 */
public class Estimation extends Sampling {

    /** Constructor. */
    public Estimation(Options o) {
        super(o);

        train = new Model(o);

        printStats(train);
        estimate(train);
        computeTheta(train);
        computePhi();

//        train.collectData("train");

//        System.out.println(computePerplexity(train));
    }

    /**
     * Gibbs sampling.
     */
    protected void sample(int m, int n) {
        int z = train.z_d[m].get(n);
        int w = train.docs.get(m).get(n);

        train.n_w_k[w][z] -= 1;
        train.n_d_k[m][z] -= 1;
        train.s_w_k[z] -= 1;
        train.s_w_d[m] -= 1;

            int K = train.labels.get(m).size();

        for (int k = 0; k < K; k++) {
            z = train.labels.get(m).get(k);

            train.p[k] = (train.n_d_k[m][z] + train.alpha) *
                    (train.n_w_k[w][z] + train.beta) /
                    (train.s_w_k[z] + train.V * train.beta);
        }
        for (int k = 1; k < K; k++) {
            train.p[k] += train.p[k - 1];
        }
        double p = Math.random() * train.p[K - 1];

        for (z = 0; z < K; z++) {
            if (train.p[z] > p)
                break;
        }
        z = train.labels.get(m).get(z);

        train.n_w_k[w][z] += 1;
        train.n_d_k[m][z] += 1;
        train.s_w_k[z] += 1;
        train.s_w_d[m] += 1;
        train.z_d[m].set(n, z);
    }

    /**
     * Topic/word distribution.
     */
    protected void computePhi() {
        for (int k = 0; k < train.K; k++) {
            for (int w = 0; w < train.V; w++) {
                train.phi[k][w] = (train.n_w_k[w][k] + train.beta) /
                                  (train.s_w_k[k] + train.V * train.beta);
            }
        }
    }

    public double[][] getPhi() {
        return train.getPhi();
    }

    public double[][] getTheta() {
        return train.getTheta();
    }
}
