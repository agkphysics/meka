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

package meka.core;

import weka.core.Attribute;
import weka.core.Instance;
import weka.core.Instances;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Comparator;
import java.util.HashMap;
import java.util.Random;
import java.util.Set;

/**
 * SuperLabelUtils.java - Handy Utils for working with Meta Labels.
 * <br>
 * TODO call this class SuperClassUtils? SuperLabelUtils? Partition? PartitionUtils?
 * @author Jesse Read 
 * @version March 2014
 */
public abstract class SuperLabelUtils {

	/**
	 * Get k subset - return a set of k label indices (of L possible labels).
	 */
	public static int[] get_k_subset(int L, int k, Random r) {
		int indices[] = A.make_sequence(L);
		A.shuffle(indices, r);
		int part[] = Arrays.copyOf(indices,k);
		Arrays.sort(part);
		return part;
	}

	private static int[][] generatePartition(int num, double M[][], Random r) {
		int L = M.length;
		int indices[] = A.make_sequence(L);

		// shuffle indices
		A.shuffle(indices,r);

		ArrayList Y_meta[] = new ArrayList[num];

		// we have a minimum of 'num' groups
		for(int i = 0; i < num; i++) {
			Y_meta[i] = new ArrayList<Integer>();
			Y_meta[i].add(indices[i]); 
		}

		// remaining
		for(int i = num; i < L; i++) {
			int idx = 0; //goesBestWith(i,Y_meta[i],M);
			Y_meta[idx].add(indices[i]);
		}

		return convertListArrayTo2DArray(Y_meta);
	}

	/**
	 * generatePartition - return [[0],...,[L-1]].
	 * @param	L	number of labels
	 * @return	[[0],...,[L-1]]
	 */
	public static int[][] generatePartition(int L) {
		int partition[][] = new int[L][];
		for(int j = 0; j < L; j++) {
			partition[j] = new int[]{j};
		}
		return partition;
	}

	/**
	 * generatePartition - .
	 * @param	indices		[1,2,..,L]
	 * @param	r			Random
	 * @return	partition
	 */
	public static int[][] generatePartition(int indices[], Random r) {
		int L = indices.length;
		return generatePartition(indices,r.nextInt(L)+1,r);
	}

	/** 
	 * Generate a random Partition.
	 * <br>
	 * TODO can generate 'indices' inside, given L
	 * <br>
	 * Get a random layout of 'num' sets of 'indices'.
	 * @param	indices		[0,1,2,...,L-1]
	 * @param	num			number of super-nodes to generate (between 1 and L)
	 * @param	r			Random, if == null, then don't randomize
	 * @return	partition
	 */
	public static int[][] generatePartition(int indices[], int num, Random r) {

		int L = indices.length;

		if (r != null)
			// shuffle indices
            A.shuffle(indices, r);

		// we have a minimum of 'num' groups
		ArrayList<Integer> selection[] = new ArrayList[num];
		for(int i = 0; i < num; i++) {
			selection[i] = new ArrayList<Integer>();
			selection[i].add(indices[i]); 
		}

		// remaining
		for(int i = num; i < L; i++) {
			int idx = r.nextInt(num);
			selection[idx].add(indices[i]);
		}

		// convert <int[]>List into an int[][] array
		int partition[][] = convertListArrayTo2DArray(selection);

		for(int part[] : partition) {
			Arrays.sort(part);
		}

		return partition;
	}


	/**
	 * Generate Random Partition
	 * @param indices	label indices
	 * @param num		the number of partitions
	 * @param	r			Random, if == null, then don't randomize
	 * @param balanced	indicate if balanced (same number of labels in each set) or not
	 * @return SORTED partition
	 */
	public static int[][] generatePartition(int indices[], int num, Random r, boolean balanced) {
		if (!balanced) 
			return generatePartition(indices,num,r);

		int L = indices.length;

		if (r != null)
			// shuffle indices
			A.shuffle(indices, r);

		int partition[][] = new int[num][];
		int k = L / num;
		int e = L % num;
		int m = 0;
		for(int c = 0; c < num; c++) {
			if (c < e) {
				partition[c] = Arrays.copyOfRange(indices,m,m+k+1);
				m = m + k + 1;
			}
			else {
				partition[c] = Arrays.copyOfRange(indices,m,Math.min(L,m+k));
				m = m + k;
			}
			Arrays.sort(partition[c]);
		}

		return partition;
	}

	/**
	 * Get Partition From Dataset Hierarchy - assumes attributes are hierarchically arranged with '.'. 
	 * For example europe.spain indicates leafnode spain of branch europe.
	 * @param	D		Dataset
	 * @return	partition
	 */
	public static final int[][] getPartitionFromDatasetHierarchy(Instances D) {
		HashMap<String,LabelSet> map = new HashMap<String,LabelSet>();
		int L = D.classIndex();
		for(int j = 0; j < L; j++) {
			String s = D.attribute(j).name().split("\\.")[0];
			LabelSet Y = map.get(s);
			if (Y==null)
				Y = new LabelSet(new int[]{j});
			else {
				Y.indices = A.append(Y.indices,j);
				Arrays.sort(Y.indices);
			}
			map.put(s, Y);
		}
		int partition[][] = new int[map.size()][];
		int i = 0;
		for(LabelSet part : map.values()) { 
			//System.out.println(""+i+": "+Arrays.toString(part.indices));
			partition[i++] = part.indices;
		}
		return partition;
	}


	/*
	 * Rating - Return a score for the super-class 'partition' using the pairwise info in 'M'.
	 * +1 if two co-ocurring labels are in different partitions. 
	 * -1 if two co-ocurring labels are in different partitions. 
	 * @param	partition	super-class partition, e.g., [[0,3],[2],[1,4]]
	 * @param	countMap	each LabelSet and its count
	public static double scorePartition(int partition[][], HashMap<LabelSet,Integer> countMap) {
		return 0.0;
	}
	public static double scorePartition(int partition[][], double M[][]) {
		return 0.0;
	}
	*/

	public static final int[][] convertListArrayTo2DArray(ArrayList<Integer> listArray[]) {
		// TODO try and do without this in the future.
		int num_partitions = listArray.length;
		int array[][] = new int[num_partitions][];
		for(int i = 0; i < listArray.length; i++) {
			array[i] = A.toPrimitive(listArray[i]); 
		}
		return array;
	}

	/**
	 * ToString - A string representation for the super-class partition 'partition'.
	 */
	public static String toString(int partition[][]) {
		StringBuilder sb = new StringBuilder();  
		sb.append("{");
		for(int i = 0; i < partition.length; i++) {
			sb.append(" "+Arrays.toString(partition[i]));
		}
		sb.append(" }");
		return sb.toString();
	}

	/**
	 * Make Partition Dataset - out of dataset D, on indices part[].
	 * @param	D		regular multi-label dataset (of L = classIndex() labels)
	 * @param	part	list of indices we want to make into an LP dataset.
	 * @return Dataset with 1 multi-valued class label, representing the combinations of part[].
	 */
	public static Instances makePartitionDataset(Instances D, int part[]) throws Exception {
		return makePartitionDataset(D,part,0,0);
	}

	/**
	 * Make Partition Dataset - out of dataset D, on indices part[].
	 * @param	D		regular multi-label dataset (of L = classIndex() labels)
	 * @param	part	list of indices we want to make into a PS dataset.
	 * @param	P		see {@link PSUtils}
	 * @param	N		see {@link PSUtils}
	 * @return Dataset with 1 multi-valued class label, representing the combinations of part[].
	 */
	public static Instances makePartitionDataset(Instances D, int part[], int P, int N) throws Exception {
		int L = D.classIndex();
		Instances D_ = new Instances(D);
		// strip out irrelevant attributes
		D_.setClassIndex(-1);
		D_ = F.keepLabels(D,L,part);
		D_.setClassIndex(part.length);
		// make LC transformation
		D_ = PSUtils.PSTransformation(D_,P,N);
		return D_;
	}

	/*
	 * Make Partition Dataset - out of dataset D, on indices part[].
	 * @param	D		regular multi-label dataset (of L = classIndex() labels)
	 * @param	part	list of indices we want to make into a PS dataset.
	 * @param	P		see {@link PSUtils}
	 * @param	N		see {@link PSUtils}
	 * @return Dataset with 1 multi-valued class label, representing the combinations of part[].
	public static Instances makePartitionDataset(Instances D, int part[], int P, int N) throws Exception {
		int L = D.classIndex();
		Instances D_ = new Instances(D);
		// strip out irrelevant attributes
		D_.setClassIndex(-1);
		D_ = F.keepLabels(D,L,part);
		D_.setClassIndex(part.length);
		// encode the relevant indices into the class attribute name
		Range r = new Range(Range.indicesToRangeList(part));
		r.setUpper(L);
		// make LC transformation
		D_ = SuperNodeFilter.mergeLabels(D_, L, "c" + r.getRanges(), P, N);
		return D_;
	}
	*/

	/**
	 * Returns a map of values for this multi-class Instances. For example, values <code>{[2,3,1], [1,0,1, [2,0,1]}</code>.
	 * @param D_	multi-class Instances
	 * @return a map where map[d] returns an int[] array of all values referred to by the d-th classification.
	 */
	public static int[][] extractValues(Instances D_) {
		int numVals = D_.classAttribute().numValues();
		int vMap[][] = new int[numVals][];
		for (int d = 0; d < numVals; d++) {
			vMap[d] = MLUtils.toIntArray(D_.classAttribute().value(d));
		}
		return vMap;
	}

	/** Decode a string into sparse list of indices */
	public static int[] decodeClass(String s) {
		return MLUtils.toIntArray(s.substring(2));
	}

	/** Encode a sparse list of indices to a string */
	public static String encodeClass(int c_[]) {
		return "c_"+(new LabelSet(c_).toString());
	}

	public static int[] decodeValue(String s) {
		return MLUtils.toIntArray(s);
	}

	/** Encode a vector of integer values to a string */
	public static String encodeValue(Instance x, int indices[]) {
		int values[] = new int[indices.length];

		for (int j = 0; j < indices.length; j++) {
			values[j] = (int)x.value(indices[j]);
		}
		return new LabelVector(values).toString();
	}

	/**
	 * GetTopNSubsets - return the top N subsets which differ from y by a single class value, ranked by the frequency storte in masterCombinations.
	 */
	public static String[] getTopNSubsets(String y, final HashMap <String,Integer>masterCombinations, int N) {
		String y_bits[] = y.split("\\+");
		ArrayList<String> Y = new ArrayList<String>();
		for(String y_ : masterCombinations.keySet()) {
			if(MLUtils.bitDifference(y_bits,y_.split("\\+")) <= 1) {
				Y.add(y_);
			}
		}
		Collections.sort(Y,new Comparator<String>(){
					public int compare(String s1, String s2) {
						// @note this is just done by the count, @todo: could add further conditions
						return (masterCombinations.get(s1) > masterCombinations.get(s2) ? -1 : (masterCombinations.get(s1) > masterCombinations.get(s2) ? 1 : 0));
					}
				}
		);
		String Y_strings[] = Y.toArray(new String[Y.size()]);
		//System.out.println("returning "+N+"of "+Arrays.toString(Y_strings));
		return Arrays.copyOf(Y_strings,Math.min(N,Y_strings.length));
	}

	/**
	 * Return a set of all the combinations of attributes at 'indices' in 'D', pruned by 'p'; AND THEIR COUNTS, e.g., {(00:3),(01:8),(11:3))}.
	 */
	public static HashMap<String,Integer> getCounts(Instances D, int indices[], int p) {
		HashMap<String,Integer> count = new HashMap<String,Integer>();
		for(int i = 0; i < D.numInstances(); i++) {
			String v = encodeValue(D.instance(i), indices);
			count.put(v, count.containsKey(v) ? count.get(v) + 1 : 1);
		}
		MLUtils.pruneCountHashMap(count,p);
		return count;
	}

	/**
	 * Super Label Transformation - transform dataset D into a dataset with <code>k</code> multi-class target attributes.
	 * Use the NSR/PS-style pruning and recomposition, according to partition 'indices', and pruning values 'p' and 'n'.
	 * @see PSUtils#PSTransformation
	 * @param indices	m by k: m super variables, each relating to k original variables
	 * @param 	D	either multi-label or multi-target dataset
	 * @param 	p	pruning value
	 * @param 	n	subset relpacement value
	 * @return	 	a multi-target dataset
	 */
	public static Instances SLTransformation(Instances D, int indices[][], int p, int n) {

		int L = D.classIndex();
		int K = indices.length;
		ArrayList<String> values[] = new ArrayList[K];
		HashMap<String,Integer> counts[] = new HashMap[K];

		// create D_
		Instances D_ = new Instances(D);

		// clear D_
		// F.removeLabels(D_,L);
		for(int j = 0; j < L; j++) {
			D_.deleteAttributeAt(0);
		}

		// create atts
		for(int j = 0; j < K; j++) {
			int att[] = indices[j];
			//int values[] = new int[2]; //getValues(indices,D,p);
			counts[j] = getCounts(D,att,p);
			Set<String> vals = counts[j].keySet(); //getValues(D,att,p);
			values[j] = new ArrayList(vals);
			D_.insertAttributeAt(new Attribute(encodeClass(att),new ArrayList(vals)),j);
		}

		// copy over values
		ArrayList<Integer> deleteList = new ArrayList<Integer>();
		for(int i = 0; i < D.numInstances(); i++) {
			Instance x = D.instance(i);
			for(int j = 0; j < K; j++) {
				String y = encodeValue(x,indices[j]);
				try {
					D_.instance(i).setValue(j,y); // y =
				} catch(Exception e) {
					// value not allowed
					deleteList.add(i); 									   // mark it for deletion
					String y_close[] = getTopNSubsets(y, counts[j], n); // get N subsets
					for(int m = 0; m < y_close.length; m++) {
						//System.out.println("add "+y_close[m]+" "+counts[j]);
						Instance x_copy = (Instance)D_.instance(i).copy();
						x_copy.setValue(j,y_close[m]);
						x_copy.setWeight(1.0/y_close.length);
						D_.add(x_copy);
					}
				}
			}
		}
		// clean up
		Collections.sort(deleteList, Collections.reverseOrder());
		//System.out.println("Deleting "+deleteList.size()+" defunct instances.");
		for (int i : deleteList) {
			D_.delete(i);
		}
		// set class
		D_.setClassIndex(K);
		// done!
		return D_;
	}

	public static double[] convertVotesToDistributionForInstance(HashMap<Integer,Double> votes[]) {
		int L = votes.length;

		double y[] = new double[L];
		for(int j = 0; j < L; j++) {
			y[j] = (Integer)MLUtils.maxItem(votes[j]);
		}
		return y;
	}

}

