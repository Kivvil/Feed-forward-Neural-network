package myNetwork;

import java.util.ArrayList;
import java.util.Random;
import java.util.concurrent.ThreadLocalRandom;

import org.apache.commons.math3.linear.ArrayRealVector;
import org.apache.commons.math3.linear.RealVector;
import org.omg.CORBA.PRIVATE_MEMBER;

import testaus.Testaus1;

public class NeuralNet {
	
	public interface Tester {
		void test(NeuralNet neuralNet);
	}
	
	private class IntervalTester {
		public IntervalTester(int batchInterval, Tester tester) {
			this.batchInterval = batchInterval;
			this.tester = tester;
		}
		public int batchInterval;
		public Tester tester;
	}
	
	public static Random random = new Random();
	
	// inputLayeria ei lasketa mukaan.
	private int layerMaara;
	
	// inputLayer mukaan.
	private int[] neuroniMaarat;
	
	// inputLayeriä ei mukaan.
	private Layer[] layerit;
	
	// Onko weight ja bias arrayhin laitettu jotain.
	private boolean isInitialized = false;
	
	private int gradientPituus;
	
	private double oppiNopeus;
	
	private ArrayList<IntervalTester> testers = new ArrayList<NeuralNet.IntervalTester>();
	
	public NeuralNet(int layerMaara, int[] neuroniMaarat, double oppiNopeus) {
		this.layerMaara = layerMaara;
		this.neuroniMaarat = neuroniMaarat;
		
		layerit = new Layer[layerMaara];
		
		int layerGradientIndex = 0;
		for(int i = 0; i < layerMaara; i++) {
			int neuronMaara = neuroniMaarat[i+1];
			int prevNeuronMaara = neuroniMaarat[i];
			layerit[i] = new Layer(neuronMaara, prevNeuronMaara, layerGradientIndex);
			layerGradientIndex += neuronMaara*prevNeuronMaara + neuronMaara + 1;
		}
		gradientPituus = layerGradientIndex;
		this.oppiNopeus = oppiNopeus;
	}
	
	/**
	 * Lisää testaajan koulutuksen ajaksi.
	 * @param batchInterval Kuinka monen batchin välein testi suoritetaan kun neuroverkkoa koulutetaan.
	 * @param tester Testaaja
	 */
	public void addTester(int batchInterval, Tester tester) {
		testers.add(new IntervalTester(batchInterval, tester));
	}

	public void randomoi(double biasMax, double biasMin, double weightMax, double weightMin) {
		for (int i = 0; i < layerit.length; i++) {
			layerit[i].randomoi(biasMax, biasMin, weightMax, weightMin);
		}
		isInitialized = true;
	}
	
	public RealVector suorita(RealVector input) throws NetworkNotInitializedException {
		if(!isInitialized) {
			throw new NetworkNotInitializedException("NeuralNetworkin weightit ja biasit ovat tyhjiä. Kutsu esim. randomoi().");
		}
		RealVector prevOutput = input;
		for (int i = 0; i < layerit.length; i++) {
			prevOutput = layerit[i].suorita(prevOutput);
		}
		return prevOutput;
	}
	
	private static int[] randArray(int min, int max, int arrayLength) {
		int[] tulos = new int[arrayLength];
		for (int i = 0; i < tulos.length; i++) {
			tulos[i] = ThreadLocalRandom.current().nextInt(min, max + 1);
		}
		return tulos;
	}
	
	/*
	 * @params:
	 * kaikkiYksilöt: array koulutuksen kaikista yksilöistä.
	 * kohteet: array indekseistä kaikissaYksilöissä, jotka otetaan miniBatchiin
	 */
	private KoulutusYksilö[] luoMiniBatch(int[] kohteet, KoulutusYksilö[] kaikkiYksilöt) {
		KoulutusYksilö[] tulos = new KoulutusYksilö[kohteet.length];
		for (int i = 0; i < tulos.length; i++) {
			tulos[i] = kaikkiYksilöt[kohteet[i]];
		}
		return tulos;
	}
	
	// Batchsize: alkaa yhdestä.
	public void kouluta(KoulutusYksilö[] yksilöt, int batchSize, int batchAmount, int repsOnBatch) throws NetworkNotInitializedException, IllegalArgumentException {
		if(batchSize > yksilöt.length) {
			throw new IllegalArgumentException("BatchSize ei voi olla suurempi kuin yksilöiden määrä.");
		}
		for(int i = 0; i < batchAmount; i++) {
			
			for (IntervalTester intervalTester : testers) {
				if ((i + 1) % intervalTester.batchInterval == 0) {
					System.out.println("Koulutetaan batch nro. " + (i+1));
					intervalTester.tester.test(this);
				}
			}

			// Generoidaan batch
			KoulutusYksilö[] batch = luoMiniBatch(randArray(0, yksilöt.length-1, batchSize), yksilöt);
			for(int repInd = 0; repInd < repsOnBatch; repInd++) {
				RealVector gradient = new ArrayRealVector(gradientPituus);
				gradient.set(0d);
				for(int yksInd = 0; yksInd < batch.length; yksInd++) {
					RealVector output = suorita(batch[yksInd].getInput());
					// cost-funktion partial derivatiivit suhteessa inputtiin
					RealVector inputDervs = new ArrayRealVector(neuroniMaarat[neuroniMaarat.length-1]);
					for(int outputInd = 0; outputInd < neuroniMaarat[neuroniMaarat.length-1]; outputInd++) {
						double derv = -2 * (output.getEntry(outputInd) - batch[yksInd].getHaluttuOutput().getEntry(outputInd));
						inputDervs.setEntry(outputInd, derv);
					}
					RealVector prevDervs = inputDervs;
					for(int layerI = layerit.length-1; layerI >= 0; layerI--) {
						prevDervs = layerit[layerI].lisääDerivatiivitGradienttiin(gradient, prevDervs);
					}
				}
				// Nyt gradientvektori on valmis, sitten kerrotaan learning ratella
				gradient.mapDivideToSelf(batch.length);
				gradient.mapMultiplyToSelf(oppiNopeus);
				
				for(int layerI = 0; layerI < layerit.length; layerI++) {
					layerit[layerI].updateUsingGradient(gradient);
				}
			}
		}
	
	}
}
