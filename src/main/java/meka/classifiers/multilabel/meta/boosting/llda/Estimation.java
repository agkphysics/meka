//import meka.classifiers.multilabel.meta.boosting.llda.Model;
//
//import java.io.*;
//
//
///**
// * Estimation of training model.
// *
// * Adapted from https://github.com/akullpp/SLDA/tree/master/Implementations/LLDA/src/main/java
// * By Flynn van Os
// */
//public class Estimation {
//    protected Model train;
//    int samples = 1;
//
//    /** Constructor. */
//    public Estimation(int numIterations) {
//
//        train = new Model(o);
//
//        printStats(train);
//        estimate(train);
//        computeTheta(train);
//        computePhi();
//        saveModel();
//
//        train.collectData("train");
//
//        System.out.println(computePerplexity(train));
//    }
//
//    /**
//     * Serialize model for loading in inference.
//     */
//    public void saveModel() {
//        FileOutputStream fos;
//
//        try {
//            fos = new FileOutputStream("training.ser");
//            ObjectOutputStream oos = new ObjectOutputStream(fos);
//            oos.writeObject(train);
//            fos.close();
//        } catch (IOException ioe) {
//            ioe.printStackTrace();
//        }
//    }
//
//    /**
//     * Gibbs sampling.
//     */
//    protected void sample(int m, int n) {
//        int z = train.z_d[m].get(n);
//        int w = train.docs.get(m).get(n);
//
//        train.n_w_k[w][z] -= 1;
//        train.n_d_k[m][z] -= 1;
//        train.s_w_k[z] -= 1;
//        train.s_w_d[m] -= 1;
//
//        int K = train.labels.get(m).size();
//
//        for (int k = 0; k < K; k++) {
//            z = train.labels.get(m).get(k);
//
//            train.p[k] = (train.n_d_k[m][z] + train.alpha) *
//                    (train.n_w_k[w][z] + train.beta) /
//                    (train.s_w_k[z] + train.V * train.beta);
//        }
//        for (int k = 1; k < K; k++) {
//            train.p[k] += train.p[k - 1];
//        }
//        double p = Math.random() * train.p[K - 1];
//
//        for (z = 0; z < K; z++) {
//            if (train.p[z] > p)
//                break;
//        }
//        z = train.labels.get(m).get(z);
//
//        train.n_w_k[w][z] += 1;
//        train.n_d_k[m][z] += 1;
//        train.s_w_k[z] += 1;
//        train.s_w_d[m] += 1;
//        train.z_d[m].set(n, z);
//    }
//
//    /**
//     * Document/topic distribution.
//     */
//    protected void computeTheta(Model model) {
//        for (int m = 0; m < model.M; m++) {
//            for (int k  = 0; k < model.K; k++) {
//                model.theta[m][k] = ((model.n_d_k[m][k] + model.alpha) /
//                        (model.s_w_d[m] + model.K * model.alpha));
//            }
//        }
//    }
//
//    /**
//     * Topic/word distribution.
//     */
//    protected void computePhi() {
//        for (int k = 0; k < train.K; k++) {
//            for (int w = 0; w < train.V; w++) {
//                train.phi[k][w] = (train.n_w_k[w][k] + train.beta) /
//                        (train.s_w_k[k] + train.V * train.beta);
//            }
//        }
//    }
//
//    protected void printStats(Model model) {
//        System.out.println(String.format("Alpha: %f\nBeta: %f\nIterations: %d\nK: %d\nV: %d\nM: %d\n",
//                model.alpha, model.beta, o.iter, model.K, model.V, model.M));
//    }
//
//    /**
//     * Gibbs sampling.
//     */
//    protected void estimate(Model model) {
//        for (int i = 1; i < o.iter; i++) {
//
//            for (int m = 0; m < model.M; m++) {
//                for (int n = 0; n < model.docs.get(m).size(); n++) {
//                    sample(m, n);
//                }
//            }
//        }
//    }
//
//    /**
//     * Computes theta and phi after a burn in period with a certain lag.
//     *
//     * @param i iteration
//     * @param model test or training model
//     */
//    protected void extended(int i, Model model) {
//        if ((i == o.iter - 1)) {
//            double Kalpha = model.K * model.alpha;
//            double Vbeta = model.V * model.beta;
//
//            for (int m = 0; m < model.M; m++) {
//                for (int k = 0; k < model.K; k++) {
//                    if (samples > 1) model.theta[m][k] *= samples - 1;
//
//                    model.theta[m][k] += ((model.n_d_k[m][k] + model.alpha) / (model.s_w_d[m] + Kalpha));
//
//                    if (samples > 1) model.theta[m][k] /= samples;
//                }
//            }
//            for (int k = 0; k < model.K; k++) {
//                for (int w = 0; w < model.V; w++) {
//                    if (samples > 1) model.phi[k][w] *= samples - 1;
//
//                    model.phi[k][w] += ((model.n_w_k[w][k] + model.beta) / (model.s_w_k[k] + Vbeta));
//
//                    if (samples > 1) model.phi[k][w] /= samples;
//                }
//            }
//            samples++;
//        }
//    }
//
//    /**
//     * Computes perplexity.
//     *
//     * Blei 2003:16 */
//    protected double computePerplexity(Model model) {
//        double loglik = 0.0;
//        int N = 0;
//
//        for (int m = 0; m < model.docs.size(); m++) {
//            for (int n = 0; n < model.docs.get(m).size(); n++) {
//                double sum = 0.0;
//                N++;
//
//                for (int k = 0; k < model.K; k++) {
//                    int w = model.docs.get(m).get(n);
//
//                    sum += model.theta[m][k] * model.phi[k][w];
//                }
//                loglik += Math.log(sum);
//            }
//        }
//        return Math.exp(-loglik / N);
//    }
//}