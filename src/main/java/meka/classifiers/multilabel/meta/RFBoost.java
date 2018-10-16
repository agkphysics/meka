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
import java.util.Enumeration;
import java.util.List;
import java.util.Vector;

import meka.classifiers.multilabel.ProblemTransformationMethod;
import meka.classifiers.multilabel.meta.boosting.llda.LLDA;
import meka.core.OptionUtils;
import weka.core.Instances;
import weka.core.Option;
import weka.core.TechnicalInformation;
import weka.core.TechnicalInformation.Field;
import weka.core.TechnicalInformation.Type;
import weka.core.TechnicalInformationHandler;

/**
 * Extremely Randomised Forest with HOMER trees algorithm.
 *
 * @author Aaron Keesing
 */
public class RFBoost extends MetaProblemTransformationMethod implements TechnicalInformationHandler {

    private static final long serialVersionUID = 2622231824645975335L;



    private int filteredFeatures;

    /**
     * Builds each HOMER tree using bagging, while randomising the settings for
     * the classifier at each node of the tree.
     */
    @Override
    public void buildClassifier(Instances D) throws Exception {
        testCapabilities(D);
        int numInstances = D.numInstances();

        // Create W, distribution matrix from LLDA
        // Create T, ranked features sorted by weights

//        m_Classifiers = ProblemTransformationMethod.makeCopies((ProblemTransformationMethod) m_Classifier, m_NumIterations);
        LLDA llda = new LLDA(0.01, 0.01, D);

        for (int i = 0; i < m_NumIterations; i++) {
            // TODO: Boosting
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
