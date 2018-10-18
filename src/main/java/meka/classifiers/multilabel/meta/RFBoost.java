/*
 *   This program is free software: you can redistribute it and/or modify
 *   it under the terms of the GNU General Public License as published by
 *   the Free Software Foundation, either version 3 of the License, or
 *   (at your option) any later version.
 *
 *   This program is distributed in the hope that it will be useful,
 *   but WITHOUT ANY WARRANTY; without even the implied warranty of
 *   MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *   GNU General Public License for more details.
 *
 *   You should have received a copy of the GNU General Public License
 *   along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

package meka.classifiers.multilabel.meta;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

import meka.classifiers.multilabel.ProblemTransformationMethod;
import meka.classifiers.multilabel.meta.boosting.llda.LLDA;
import meka.core.OptionUtils;
import weka.core.Instance;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;

/**
 * RFBoost
 *
 * @author Flynn van Os
 */
public class RFBoost extends ProblemTransformationMethod implements TechnicalInformationHandler {

    private static final long serialVersionUID = 2622231824645975335L;

    private double[][][] hypotheses;
    private int[] featuresForHypotheses;

    private int filteredFeatures = 50;

    private int m_NumIterations = 200;

    /**
     * Builds RFBoost hypotheses
     */
    @Override
    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        int numInstances = D.numInstances();
        int numFeatures = D.numAttributes() - D.classIndex();

        // Create W, distribution matrix from LLDA
        // Create T, ranked features sorted by weights

//        m_Classifiers = ProblemTransformationMethod.makeCopies((ProblemTransformationMethod) m_Classifier, m_NumIterations);
        LLDA llda = new LLDA(0.5, 0.1, D);
        double[][] W = llda.getTheta();
//        LLDAalt llda2 = new LLDAalt(0.5, 0.1, D);
//        double[][] W = llda2.getTheta();

        // ALT
//        double[][] W_alt = W.clone();
//        for (int i = 0; i < W.length; i++) {
//            for (int j = 0; j < W[0].length; j++) {
//                W_alt[i][j] = 1.0 / (D.classIndex() * numInstances);
//            }
//        }
//        W = W_alt;

        // END ALT

        normalize(W, D.size());

        List<Integer> rankedFeatures = rankFeatures(D, llda.getPhi());

        hypotheses = new double[m_NumIterations][2][D.classIndex()];
        featuresForHypotheses = new int[m_NumIterations];

        List<Integer> filteredList = new ArrayList<>(rankedFeatures.subList(0, filteredFeatures)); // TODO: Select top features?
        int offset = filteredFeatures;

        double epsilon = 1.0 / (D.classIndex() * numInstances);

        for (int iteration = 0; iteration < m_NumIterations; iteration++) {
            // TODO: Boosting
            int pivot_term = filteredList.get(0); // TODO: Correct?
            int pivot_index = 0;
            double z_min = Integer.MAX_VALUE;

            double[][] c_u_l = new double[2][D.classIndex()];

            for (int j = 0; j < filteredFeatures; j++) {
                int term_k = filteredList.get(j);

                // Calculate Wul + or -
                double[] W_0_plus_l = new double[D.classIndex()];
                double[] W_1_plus_l = new double[D.classIndex()];
                double[] W_0_min_l = new double[D.classIndex()];
                double[] W_1_min_l = new double[D.classIndex()];
                for (int i = 0; i < D.size(); i++) {
                    Instance doc = D.get(i);
                    // TODO: New method

                    // Equations 10 and 11
                    for(int l = 0; l < D.classIndex(); l++) {
                        if (doc.value(term_k) == 1.0) {
                            if (doc.value(l) == 1.0) {
                                W_1_plus_l[l] = W[i][l];
                            } else {
                                W_1_min_l[l] = W[i][l];
                            }
                        } else {
                            if (doc.value(l) == 1.0) {
                                W_0_plus_l[l] = W[i][l];
                            } else {
                                W_0_min_l[l] = W[i][l];
                            }
                        }
                    }
                }

                // Equation 13 is all below
                double label_sum = 0;
                for (int l = 0; l < D.classIndex(); l++) {
                    label_sum = label_sum + Math.sqrt(W_0_min_l[l] * W_0_plus_l[l]);
                    label_sum = label_sum + Math.sqrt(W_1_min_l[l] * W_1_plus_l[l]);
                }
                double z_k_r = 2 * label_sum;

                // See algorithm 4 lines 8 - 13
                if (z_k_r < z_min) {
                    pivot_term = filteredList.get(j);
                    pivot_index = j;
                    for (int l = 0; l < D.classIndex(); l++) {
                        c_u_l[0][l] = 0.5 * Math.log((W_0_plus_l[l] + epsilon) / (W_0_min_l[l] + epsilon));
                        c_u_l[1][l] = 0.5 * Math.log((W_1_plus_l[l] + epsilon) / (W_1_min_l[l] + epsilon));
                    }
                    z_min = z_k_r;
                }
            }

            // Build weak hypothesis
            // Equation 8 + Algorithm 4 line 13
            hypotheses[iteration] = c_u_l; // Weak hypothsis?
            featuresForHypotheses[iteration] = pivot_term;

            // Update weights
            // Equation 4 + Algorithm 4 line 14
            for (int i = 0; i < W.length; i++) {
                for (int l = 0; l < D.classIndex(); l++) {
                    double prevWeight = W[i][l];
                    double hypothesis;
                    if (D.get(i).value(pivot_term) == 1.0) {
                        hypothesis = c_u_l[1][l];
                    } else {
                        hypothesis = c_u_l[0][l];
                    }

                    int phi_val = targetFunction(D.get(i).value(l));
                    int hyp_val = targetFunction(hypothesis);

                    W[i][l] = (W[i][l] * Math.exp(-1 * phi_val * hyp_val)) / z_min;
                }
            }

            // Algorithm 4 line 15 + 16
            filteredList.remove(pivot_index);
            filteredList.add(rankedFeatures.get(offset));
            offset = (offset + 1) % numFeatures;
        }
    }

    private int targetFunction(double value) {
        return value > 0 ? 1 : -1;
    }


    @Override
    public double[] distributionForInstance(Instance x) throws Exception {
        // Equation 7
        double[] result = new double[x.classIndex()];
        for (int r = 0; r < m_NumIterations; r++) {
            double[][] hypothesis = hypotheses[r];
            int feature = featuresForHypotheses[r];
            for (int l = 0; l < x.classIndex(); l++) {
                if (x.value(feature) == 1.0) {
                    result[l] = result[l] + hypothesis[1][l];
                } else {
                    result[l] = result[l] + hypothesis[0][l];
                }
            }
        }
        // Equation 5
        for (int i = 0; i < result.length; i++) {
            result[i] = result[i] >= 0 ? 1 : 0;
        }
        return result;
    }

    private List<Integer> rankFeatures(Instances D, double[][] phi) {
        int numFeatures = D.numAttributes() - D.classIndex();
        List<Integer> result = new ArrayList<>(numFeatures);


        List<Double> phiMaxList = new ArrayList<>(numFeatures);
        for (int i = 0; i < numFeatures; i++) {
            double phi_max_tk = phi[0][i];
            for (int l = 1; l < D.classIndex(); l++) {
                if (phi[l][i] > phi_max_tk) {
                    phi_max_tk = phi[l][i];
                }
            }
            result.add(i + D.classIndex());
            phiMaxList.add(phi_max_tk);
        }
        result.sort(new Comparator<Integer>() {
            @Override public int compare(Integer o1, Integer o2) {
                return Double.compare(phiMaxList.get(o2 - D.classIndex()), phiMaxList.get(o1 - D.classIndex()));
            }
        });

        return result;
    }

    private void normalize(double[][] W, int n) {
        for (int i = 0; i < W.length; i++) {
            for (int j = 0; j < W[i].length; j++) {
                W[i][j] = W[i][j] / n;
            }
        }
    }

    @Override
    public String[] getOptions() {
        List<String> result = new ArrayList<>();
        OptionUtils.add(result, "f", filteredFeatures);
        OptionUtils.add(result, super.getOptions());
        return OptionUtils.toArray(result);
    }

    @Override
    public void setOptions(String[] options) throws Exception {
        super.setOptions(options);
        this.filteredFeatures = OptionUtils.parse(options, "f", 10);
    }

    @Override
    public Enumeration<Option> listOptions() {
        Vector<Option> options = new Vector<>();
        options.add(new Option("Number of features to filter", "filteredFeatures", 1, "-f filtered"));
        OptionUtils.add(options, super.listOptions());
        return options.elements();
    }

    @Override
    public String globalInfo() {
        return "Extremely Randomised Forest of HOMER trees.";
    }

    @Override
    public TechnicalInformation getTechnicalInformation() {
        TechnicalInformation info = new TechnicalInformation(Type.INPROCEEDINGS);
        info.setValue(Field.AUTHOR, "Al-Salemi, Bassam and Noah, S.A.M. and Ab Aziz, Mohd Juzaiddin");
//        info.setValue(Field.EDITOR, "Sun, Yi and Lu, Huchuan and Zhang, Lihe and Yang, Jian and Huang, Hua");
        info.setValue(Field.TITLE, "RFBoost: An improved multi-label boosting algorithm and its application to text categorisation");
        info.setValue(Field.BOOKTITLE, "Knowledge-Based Systems");
        info.setValue(Field.YEAR, "2016");
//        info.setValue(Field.PUBLISHER, "Springer International Publishing");
//        info.setValue(Field.ADDRESS, "Cham");
//        info.setValue(Field.PAGES, "450--460");
//        info.setValue(Field.ISBN, "978-3-319-67777-4");
        return info;
    }

    public static void main(String[] args) {
        ProblemTransformationMethod.runClassifier(new RFBoost(), args);
    }
}
