package meka.classifiers.multilabel.meta.boosting.llda; /**
 Following JGibbLabeledLDA and JGibbLDA, this code is licensed under the GPLv2.
 Please see the LICENSE file for the full license.
 */

import weka.core.Instances;

/** Main. */
public class LLDA {

    private Estimation estimation;

    public LLDA(double alpha, double beta, Instances D) {
        Options options = new Options(D, alpha, beta);
        estimation = new Estimation(options);
    }

    public double[][] getPhi() {
        return estimation.getPhi();
    }

    public double[][] getTheta() {
        return estimation.getTheta();
    }


}
