for d in 11 23; do 
	for eps in 0.05 0.025 0.01; do 
		for run in 2; do 
			python plot.py --eps $eps \
				--run $run \
				--sample $d: \
				--truncate 10000000 \
				--omit_beginning 1000 \
				--outfile $d-$eps.png; 
		done; 
	done; 
done
