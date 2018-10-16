package meka.classifiers.multilabel.meta.boosting.llda;

import java.io.*;
import java.util.*;

import weka.core.Instance;
import weka.core.Instances;


public class Model implements Serializable {
    public transient Options options;

    // docs[m][n] = document m word n
    public ArrayList<ArrayList<Integer>> docs;
    // labels[m] = labeleset k for document m
    public ArrayList<ArrayList<Integer>> labels;
    // number of documents
    public int M;
    // number of types
    public int V;
    // number of topics
    public int K;
    // dirichlet alpha for theta
    public double alpha;
    // asymmetric alphas for estimation, length K
    public double[] alphas;
    // dirichlet beta for phi
    public double beta;
    // theta[m][k] = probability of topic k in document m
    public double[][] theta;
    // phi[k][n] = probability of word n in topic k
    public double[][] phi;
    // n_w_k[w][k] = counts for word n under topic k
    public int[][] n_w_k;
    // n_d_k[m][k] = counts for topic k in document m
    public int[][] n_d_k;
    // s_w_k[k] = count of word n under topic k
    public int[] s_w_k;
    // s_w_d[m] = count of words n in document m
    public int[] s_w_d;
    // p[k] = probability of topic k
    public double[] p;
    // z_d[m] = topic assignments in document m
    public ArrayList<Integer>[] z_d;

    /** Constructor.
     *
     * @param options options
     * */
    public Model(Options options) {
        this(options, null);
    }

    /**
     * Constructor for testing.
     *
     * @param options options
     * @param train trainings model
     */
    public Model(Options options, Model train) {
        this.options = options;
        this.alpha = options.alpha;
        this.beta = options.beta;
        this.K = options.K;

        docs = new ArrayList<ArrayList<Integer>>();

        labels = new ArrayList<ArrayList<Integer>>();
        readCorpus(options.corpus);
        initModel();
    }

    /** Initializes datastructures. */
    public void initModel() {
        M = docs.size();
        p = new double[K];
        theta = new double[M][K];
        phi = new double[K][V];
        n_w_k = new int[V][K];
        n_d_k = new int[M][K];
        s_w_k = new int[K];
        s_w_d = new int[M];
        z_d = new ArrayList[M];

        initTopics();
    }

    /** Initializes topics randomly. */
    public void initTopics() {
        Random r = new Random();

        for (int m = 0; m < M; m ++) {
            z_d[m] = new ArrayList<Integer>();
            ArrayList<Integer> doc = docs.get(m);
            int N = doc.size();

            for (int n = 0; n < N; n++) {
                int k = r.nextInt(K);
                z_d[m].add(k);
                n_w_k[doc.get(n)][k]++;
                n_d_k[m][k]++;
                s_w_k[k]++;
            }
            s_w_d[m] = N;
        }
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

    /** Read the training or testing corpus. */
    public void readCorpus(Instances corpus) {
        String line;
        int lll = 0;

        this.M = corpus.size();

        for (int i = 0; i < M; i++) {
            Instance doc = corpus.get(i);
            int docLength = doc.numValues();
            ArrayList<Integer> labels = new ArrayList<>();
            ArrayList<Integer> words = new ArrayList<>();
            extractLabelsTopics(doc, labels, words);
            this.labels.add(labels);
            this.docs.add(words);
        }

        this.K = corpus.classIndex();
        this.V = corpus.numAttributes() - corpus.classIndex(); // I think?
    }

    public double[][] getPhi() {
        return this.phi;
    }

    public double[][] getTheta() {
        return this.theta;
    }
}
