package myNetwork;

import java.util.ArrayList;
import java.util.List;

public class ThreadPool {
	private List<Thread> jono = new ArrayList<Thread>();
	private int maxThredit;
	
	public ThreadPool(int maxThredit) {
		this.maxThredit = maxThredit;
	}
	
	public void lisaaThread(final Thread t ) {
		jono.add(t);
	}
	
	public void aloita() {
	}
}
