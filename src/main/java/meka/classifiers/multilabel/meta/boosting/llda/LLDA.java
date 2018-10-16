package meka.classifiers.multilabel.meta.boosting.llda; /**
 Following JGibbLabeledLDA and JGibbLDA, this code is licensed under the GPLv2.
 Please see the LICENSE file for the full license.
 */


import java.util.ArrayList;
import java.util.List;

import meka.core.A;
import meka.core.F;
import weka.core.Instance;
import weka.core.Instances;

/** Main. */
public class LLDA {

    private float alpha;
    private float beta;
    private int K;

    private int[] labels;

    private float[] z_m_n;
    private int[][] n_m_z;
    private int[][] n_z_t;
    private int[][] n_z;

    public LLDA(float alpha, float beta, Instances D) {
        this.K = D.classIndex();
        this.alpha = alpha;
        this.beta = beta;
        labels = A.make_sequence(K);
    }
}
