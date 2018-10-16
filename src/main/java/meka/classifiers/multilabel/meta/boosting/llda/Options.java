package meka.classifiers.multilabel.meta.boosting.llda;


import weka.core.Instances;

/** Commandline options */
public class Options {
    public boolean estimation = true;

    //Corpus, TODO: Not from file
    public Instances corpus;

    //Alpha
    public double alpha = 0.01;

    //Beta
    public double beta = 0.01;

    //Iterations
    public int iter = 1000;

    //Topics
    public int K = 3;

    public boolean llda = true;

    //Reporting step
    public int step = 100;

    public int burn = 100;

    public int lag = 5;

    public String method = "extended";

    public boolean bg = true;
}
