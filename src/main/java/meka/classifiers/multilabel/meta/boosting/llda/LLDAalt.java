package meka.classifiers.multilabel.meta.boosting.llda; /**
 Following JGibbLabeledLDA and JGibbLDA, this code is licensed under the GPLv2.
 Please see the LICENSE file for the full license.
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.HashSet;
import java.util.List;
import java.util.Random;
import java.util.Set;
import java.util.stream.DoubleStream;
import java.util.stream.IntStream;

import weka.core.Instance;
import weka.core.Instances;

/** Main. */
public class LLDAalt {

    private int K;
    private int M;
    private int V;
    private Random random;
    private int iterations = 50;

    private List<List<Integer>> z_m_n;
    private int[][] n_m_z;
    private int[][] n_z_t;
    private int[] n_z;
    private List<List<Integer>> labelsForDoc;
    private List<List<Integer>> wordsForDoc;
    private double alpha;
    private double beta;

    public LLDAalt(double alpha, double beta, Instances D) {
        this.alpha = alpha;
        this.beta = beta;
        random = new Random();
        readCorpus(D);
        for (int i = 0; i < iterations; i++) {
            inference();
        }
    }

    /** Read the corpus. */
    public void readCorpus(Instances docs) {
        this.M = docs.size();
        this.K = docs.classIndex();
        this.V = docs.numAttributes() - docs.classIndex();
        initArrays();

        for (int m = 0; m < M; m++) {
            Instance doc = docs.get(m);
            ArrayList<Integer> labels = new ArrayList<>();
            ArrayList<Integer> words = new ArrayList<>();

            extractLabelsTopics(doc, labels, words);
            this.labelsForDoc.add(labels);
            this.wordsForDoc.add(words);
            int docLength = words.size();

            List<Integer> z_n = new ArrayList<>(docLength);
            for (int j = 0; j < docLength; j++) {
                int label = random.nextInt(labels.size());
                z_n.add(label);
            }
            z_m_n.add(z_n);
            for (int j = 0; j < docLength; j++) {
                int t = words.get(j);
                int z = z_n.get(j);
                n_m_z[m][z]++;
                n_z_t[z][t]++;
                n_z[z]++;
            }

        }
    }

    private void initArrays() {
        this.z_m_n = new ArrayList<>(this.M);
        this.n_m_z = new int[this.M][this.K];
        this.n_z_t = new int[this.K][this.V];
        this.n_z = new int[this.K];
        this.labelsForDoc = new ArrayList<>(this.M);
        this.wordsForDoc = new ArrayList<>(this.M);
    }

    private void inference() {
        for (int m = 0; m < wordsForDoc.size(); m++) {
            List<Integer> words = wordsForDoc.get(m);
            List<Integer> labels = labelsForDoc.get(m);
            for (int n = 0; n < words.size(); n++) {
                int t = words.get(n);
                int z = z_m_n.get(m).get(n);

                n_m_z[m][z]--;
                n_z_t[z][t]--;
                n_z[z]--;

                double denom_a = IntStream.of(this.n_m_z[m]).sum() + this.K * this.alpha;
//                double denom_a = IntStream.of(this.n_m_z[m]).sum() + this.K * this.bew;

                double[] p_z = new double[labels.size()];
                double sum = 0;
                for (int i = 0; i < p_z.length; i++) {
                    int label = labels.get(i);
                    double nztting = this.n_z_t[i][t] + this.beta;
                    double denom_b = IntStream.of(this.n_z_t[i]).sum() + this.V * this.beta;
                    double nmzting = this.n_m_z[m][i] + this.alpha;
                    p_z[i] = label * nztting / denom_b * nmzting / denom_a;
                    sum = sum + p_z[i];
                }
                //normalize
                for (int i = 0; i < p_z.length; i++) {
                    p_z[i] = p_z[i] / sum;
                }
                int winner = sampleMultiNomial(p_z);
                this.z_m_n.get(m).set(n, winner);
                this.n_m_z[m][winner]++;
                this.n_z_t[winner][t]++;
                this.n_z[winner]++;
            }
        }
    }

    private int sampleMultiNomial(double[] p_z) {
        Arrays.sort(p_z);
        double gen = random.nextDouble();
        double prev = 0;
        for (int i = 0; i < p_z.length; i++) {
            double comp = p_z[i] + prev;
            if (gen >= prev && gen <= comp) {
                return i;
            }
            prev = comp;
        }
        return p_z.length - 1;
    }

    public double[][] getPhi() {
        double[][] phi = new double[K][V];
//        for (int k = 0; k < this.K; k++) {
//            for (int v = 0; v < this.V; v++) {
//                phi[k][v] = (this.n_z_t[k][v] + this.beta) /
//                        (this.n_z[k] + this.V * this.beta);
//            }
//        }

        for (int k = 0; k < K; k++) {
            for (int v = 0; v < V; v++) {
                phi[k][v] = (n_z_t[k][v] + this.beta) / (this.n_z[k] + this.V * this.beta);
            }
        }
        return phi;
    }

    public double[][] getTheta() {
        double[][] theta = new double[M][K];
        double[][] n_alpha = new double[M][K];
        for (int m = 0; m < M; m++) {
            Set<Integer> labels = new HashSet<>(labelsForDoc.get(m));
            for (int k = 0; k < K; k++) {
                double added = labels.contains(k) ? alpha : 0;
                n_alpha[m][k] = n_m_z[m][k] +added;
            }
        }
        for (int m = 0; m < M; m++) {
            double rowSum = DoubleStream.of(n_alpha[m]).sum();
            for (int k = 0; k < K; k++) {
                theta[m][k] = n_alpha[m][k] / rowSum;
            }
        }
        return theta;
    }


    private void extractLabelsTopics(Instance doc, List<Integer> labels, List<Integer> words) {
        int L = doc.classIndex();
        for (int i = 0; i < doc.numValues(); i++) {
            int value = doc.index(i);
            if (value < L) {
                labels.add(value);
            } else {
                words.add(value - L);
            }
        }
    }


}
