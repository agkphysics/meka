package meka.classifiers.multilabel.meta.boosting.llda;


import weka.core.Instances;

/** Commandline options */
public class Options {

    public Options(Instances corpus, double alpha, double beta) {
        this.corpus = corpus;
        this.alpha = alpha;
        this.beta = beta;
    }

    //Corpus, TODO: Not from file
    public Instances corpus;

    //Alpha
    public double alpha = 0.5;

    //Beta
    public double beta = 0.1;

    //Iterations
    public int iter = 50;

    //Topics
    public int K = 3;

    public int burn = 100;

    public int lag = 5;

    public String method = "simple";


}
