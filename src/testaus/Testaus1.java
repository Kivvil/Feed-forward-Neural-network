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

public class Testaus1 implements NeuralNet.Tester {
	
	private static final double BIAS_MAX = 1.75, BIAS_MIN = -1.75;
	private static final double WEIGHT_MAX = 1.75, WEIGHT_MIN = -1.75;
	private static final int TEST_BATCH_INTERVAL = 5;
	
	private static final int LAYER_MAARA = 3;
	private static final int[] NEURON_MAARAT = {
			784, 30, 50, 10
	};
	private static final double OPPINOPEUS = 0.1;
	private static final int BATCH_SIZE = 50;
	private static final int REPS_ON_BATCH = 100;
	private static final int BATCH_MAARA = 500;
	
	private MnistMatrix[] mnistMatrix;
	private KoulutusYksilö[] yksilöt;
	private KoulutusYksilö[] testiyksilöt;
	private MnistMatrix[] testimnistMatrix;
	
	public void aloita() {
		NeuralNet net = new NeuralNet(LAYER_MAARA, NEURON_MAARAT, OPPINOPEUS);
		net.addTester(TEST_BATCH_INTERVAL, this);
		net.randomoi(BIAS_MAX, BIAS_MIN, WEIGHT_MAX, WEIGHT_MIN);
		try {
			mnistMatrix = new MnistDataReader().readData(
					"/home/vilho/eclipse-workspace/NeuralNetwork2/testResources/train-images.idx3-ubyte", "/home/vilho/eclipse-workspace/NeuralNetwork2/testResources/train-labels.idx1-ubyte");
			testimnistMatrix = new MnistDataReader().readData(
					"/home/vilho/eclipse-workspace/NeuralNetwork2/testResources/t10k-images.idx3-ubyte", "/home/vilho/eclipse-workspace/NeuralNetwork2/testResources/t10k-labels.idx1-ubyte");
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
		testiyksilöt = new KoulutusYksilö[testimnistMatrix.length];
		for (int i = 0; i < testimnistMatrix.length; i++) {
			MnistMatrix matrix = testimnistMatrix[i];
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
			testiyksilöt[i] = new KoulutusYksilö(input, haluttuOutput);
		}
		try {
			net.kouluta(yksilöt, BATCH_SIZE, BATCH_MAARA, REPS_ON_BATCH);
		} catch (IllegalArgumentException | NetworkNotInitializedException e) {
			e.printStackTrace();
		}
		test(net);
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

	@Override
	public void test(NeuralNet neuralNet) {
		int oikein = 0;
		for(int i = 0; i < testiyksilöt.length; i++) {
			RealVector output = null;
			try {
				output = neuralNet.suorita(testiyksilöt[i].getInput());
			} catch (NetworkNotInitializedException e) {
				e.printStackTrace();
			}
			if(output.getMaxIndex() == testiyksilöt[i].getHaluttuOutput().getMaxIndex()) {
				oikein++;
			}
		}
		System.out.println("Oikein testissä meni " + oikein);
		float prosentti = ((float)oikein) / ((float)testiyksilöt.length)  * 100f;
		System.out.println("Prosentti: " + prosentti);
		System.out.println("--------------------------------------------------------");
	}

	public static void main(String[] args) {
		Testaus1 tester = new Testaus1();
		tester.aloita();
	}
}
