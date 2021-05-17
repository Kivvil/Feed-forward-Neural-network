package myNetwork;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.MatrixUtils;
import org.apache.commons.math3.linear.RealMatrix;
import org.apache.commons.math3.linear.RealVector;

public class Layer {
	// Layerin neuronejen määrä.
	private int neuronMaara;
	// Edellisen layerin neuronejen määrä.
	private int prevNeuronMaara;
	
	// Weightit edellisen layerin neuroneihin.
	// Rowmaara = neuronMaara, Colmaara = prevNeuronMaara
	private RealMatrix weights;
	
	private int weightMaara;
	
	/// Tämän layerin jokaisen layerin bias-arvo. (se arvo, joka lisätään sigmoidin parametriin)
	private RealVector biases;
	
	// Indeksi, jossa tämän layerin jutut alkavat gradientvektorissa (inclusive).
	private int gradientIndeksi;
	
	// Indeksi, jossa tämän layerin jutut lpppuvat gradientvektorissa (exclusive). 
	private int gradientIndeksiLoppu;
	
	// Viimesuoritukset aktivaatiot
	private RealVector activations;
	
	private RealVector lastLayerActivations;

	public Layer(int neuronMaara, int prevNeuronMaara, int gradientIndeksi) {
		this.neuronMaara = neuronMaara;
		this.prevNeuronMaara = prevNeuronMaara;
		weights = MatrixUtils.createRealMatrix(neuronMaara, prevNeuronMaara);
		biases = new ArrayRealVector(neuronMaara);
		weightMaara = neuronMaara * prevNeuronMaara;
		this.gradientIndeksi = gradientIndeksi;
		gradientIndeksiLoppu = gradientIndeksi + weightMaara + /*biasit:*/ neuronMaara;
	}
	
	public void randomoi(double biasMax, double biasMin, double weightMax, double weightMin) {
		// Randomoidaan weight matrix.
		for (int row = 0; row < weights.getRowDimension(); row++) {
			for (int col = 0; col < weights.getColumnDimension(); col++) {
				double weight = (weightMax - weightMin) * NeuralNet.random.nextDouble() + weightMin;
				weights.setEntry(row, col, weight);
			}
		}
		
		// Randomoidaan biasit.
		for (int i = 0; i < biases.getDimension(); i++) {
			double bias = (biasMax - biasMin) * NeuralNet.random.nextDouble() + biasMin;
			biases.setEntry(i, bias);
		}
	}
	
	/*
	 * @params
	 * 	prevActs: edellisten neuronejen aktivaatiot
	 * 
	 * def:
	 * 	ns. Feed forwardataan tämän layerin läpi
	 */
	public RealVector suorita(RealVector prevActs) {
		RealVector weightedActs = weights.operate(prevActs);
		RealVector biased = weightedActs.add(biases);
		activations = sigmoid(biased);
		lastLayerActivations = prevActs;
		return activations;
	}
	
	private static RealVector sigmoid(RealVector x) {
		RealVector tulos = x.copy();
		for (int i = 0; i < tulos.getDimension(); i++) {
			tulos.setEntry(i, sigmoid(tulos.getEntry(i)));
		}
		return tulos;
	}
	
	private static double sigmoid(double x) {
		return 1 / (1 + Math.exp(-x));
	}

	public int getNeuronMaara() {
		return neuronMaara;
	}

	public int getWeightMaara() {
		return weightMaara;
	}

	public void setWeights(RealMatrix weights) {
		this.weights = weights;
	}

	public void setBiases(RealVector biases) {
		this.biases = biases;
	}
	
	// gradient täytyy olla jo kerrottu learning ratella.
	public void updateUsingGradient(RealVector gradient) {
		//System.out.println("Col dimension: " + weights.getColumnDimension());
		//System.out.println("Row dimension: " + weights.getRowDimension());
		for (int i = gradientIndeksi; i < gradientIndeksiLoppu; i++) {
			int indeksiLayerissa = i - gradientIndeksi;
			if(i < gradientIndeksi + weightMaara) {
				// meneillään weightit
				int row = indeksiLayerissa / weights.getColumnDimension();
				//System.out.println("row:" + row + " / index:" + indeksiLayerissa);
				int col = indeksiLayerissa - row*(weights.getColumnDimension());
				weights.addToEntry(row, col, gradient.getEntry(i));
			} else {
				// Meneillään biasit
				int biasIndex = indeksiLayerissa - weightMaara;
				biases.addToEntry(biasIndex, gradient.getEntry(i));
				
			}
		}
	}
	
	// Palauttaa neuronejen edellisten(a.k.a. seuraavien backpropagationissa) neuronejen vaikutuksen.
	// Huom: nextNeuron a.k.a. prosessissa(backpropagation) edellinen/previous
	public RealVector lisääDerivatiivitGradienttiin(RealVector gradient, RealVector neuronDervs) {
		// Laitetaan weightit
		
		for (int row = 0; row < weights.getRowDimension(); row++) {
			double derAinZ = derivOfSigmoidResult(activations.getEntry(row));
			for (int col = 0; col < weights.getColumnDimension(); col++) {
				double derZinW = lastLayerActivations.getEntry(col);
				double derv = derZinW * derAinZ * neuronDervs.getEntry(row);
				//System.out.println("Weightderv: " + derv);
				int weightIndexInGradient = gradientIndeksi + row * weights.getColumnDimension() + col;
				gradient.addToEntry(weightIndexInGradient, derv);
			}
			// ROW on myös tämän layerin neuroniIndeksi a.k.a. biasIndeksi.
			double biasDerv = derAinZ * neuronDervs.getEntry(row);
			//System.out.println("Biasderv: " + biasDerv);
			int biasIndex = gradientIndeksi + weightMaara + row;
			gradient.addToEntry(biasIndex, biasDerv);
		}
		
		RealVector seuraavaLayerActDervs = new ArrayRealVector(prevNeuronMaara);
		
		// Lasketaan prosessissa seuraavan layerin neuronejen vaikututs c:hen
		for(int prevNeuronI = 0; prevNeuronI < prevNeuronMaara; prevNeuronI++) {
			double prevNeuronDerv = 0.0;
			for(int thisNeuronI = 0; thisNeuronI < neuronMaara; thisNeuronI++) {
				double derZinAl_1 = weights.getEntry(thisNeuronI, prevNeuronI);
				double derAinZ = derivOfSigmoidResult(activations.getEntry(thisNeuronI));
				prevNeuronDerv += derZinAl_1 * derAinZ * neuronDervs.getEntry(thisNeuronI);
			}
			seuraavaLayerActDervs.setEntry(prevNeuronI, prevNeuronDerv);
		}
		
		return seuraavaLayerActDervs;
	}
	
	private static double derivOfSigmoidResult(double sigResult) {
		return sigResult * (1 - sigResult);
	}
	
}
