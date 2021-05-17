package testaus;

import java.io.IOException;
import java.util.Arrays;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

import mnistDataReader.MnistDataReader;
import mnistDataReader.MnistMatrix;
import myNetwork.KoulutusYksilö;
import myNetwork.Layer;
import myNetwork.NetworkNotInitializedException;
import myNetwork.NeuralNet;

public class Testaus1 {
	
	private static final double BIAS_MAX = 1.75, BIAS_MIN = -1.75;
	private static final double WEIGHT_MAX = 1.75, WEIGHT_MIN = -1.75;
	
	private static final int LAYER_MAARA = 2;
	private static final int[] NEURON_MAARAT = {
			784, 10, 10
	};
	private static final double OPPINOPEUS = 0.1;
	private static final int BATCH_SIZE = 20;
	private static final int REPS_ON_BATCH = 100;
	private static final int BATCH_MAARA = 100000;
	
	private static NeuralNet net;
	private static MnistMatrix[] mnistMatrix;
	private static KoulutusYksilö[] yksilöt;

	public static void main(String[] args) {
		net = new NeuralNet(LAYER_MAARA, NEURON_MAARAT, OPPINOPEUS);
		net.randomoi(BIAS_MAX, BIAS_MIN, WEIGHT_MAX, WEIGHT_MIN);
		try {
			mnistMatrix = new MnistDataReader().readData(
					"/home/vilho/eclipse-workspace/NeuralNetwork2/testResources/train-images.idx3-ubyte", "/home/vilho/eclipse-workspace/NeuralNetwork2/testResources/train-labels.idx1-ubyte");
		} catch (IOException e) {
			e.printStackTrace();
			System.exit(-1);
		}
		//printMnistMatrix(mnistMatrix[0]);
		//System.exit(0);
		yksilöt = new KoulutusYksilö[mnistMatrix.length];
		for (int i = 0; i < yksilöt.length; i++) {
			MnistMatrix matrix = mnistMatrix[i];
			RealVector haluttuOutput = new ArrayRealVector(10);
			haluttuOutput.set(0);
			haluttuOutput.setEntry(matrix.getLabel(), 1);
			RealVector input = new ArrayRealVector(784);
			int totInd = 0;
			for(int row = 0; row < matrix.getNumberOfRows(); row++) {
				for(int col = 0; col < matrix.getNumberOfColumns(); col++) {
					input.setEntry(totInd, matrix.getValue(row, col) / 255d);
					totInd++;
				}
			}
			yksilöt[i] = new KoulutusYksilö(input, haluttuOutput);
		}
		try {
			net.kouluta(yksilöt, BATCH_SIZE, BATCH_MAARA, REPS_ON_BATCH);
		} catch (IllegalArgumentException | NetworkNotInitializedException e) {
			e.printStackTrace();
		}
		testaa();
	}
	
	public static void testaa() {
		int oikein = 0;
		for(int i = 0; i < 1000; i++) {
			RealVector output = null;
			try {
				output = net.suorita(yksilöt[i].getInput());
			} catch (NetworkNotInitializedException e) {
				e.printStackTrace();
			}
			if(output.getMaxIndex() == yksilöt[i].getHaluttuOutput().getMaxIndex()) {
				oikein++;
			}
		}
		System.out.println("Oikein testissä meni " + oikein);
		float prosentti = ((float)oikein) / 1000f  * 100f;
		System.out.println("Prosentti: " + prosentti);		
	}
	
	private static void printMnistMatrix(final MnistMatrix matrix) {
        System.out.println("label: " + matrix.getLabel());
        for (int r = 0; r < matrix.getNumberOfRows(); r++ ) {
            for (int c = 0; c < matrix.getNumberOfColumns(); c++) {
                System.out.print(matrix.getValue(r, c) + " ");
            }
            System.out.println();
        }
    }

}
