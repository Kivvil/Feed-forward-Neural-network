package myNetwork;

import org.apache.commons.math3.linear.RealVector;

public class KoulutusYksilö {
	private RealVector input;
	private RealVector haluttuOutput;
	
	public KoulutusYksilö(RealVector input, RealVector haluttuOutput) {
		super();
		this.input = input;
		this.haluttuOutput = haluttuOutput;
	}

	public RealVector getInput() {
		return input;
	}

	public RealVector getHaluttuOutput() {
		return haluttuOutput;
	}
}
