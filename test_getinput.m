function inp = test_getinput(image,meta,buckets)

	audfile 	= [image(1:end-3),'wav'];
	z         	= audioread(audfile);
    
    
    
    
    
	SPEC 		= runSpec(z,meta.audio);
	mu    		= mean(SPEC,2);
    stdev 		= std(SPEC,[],2) ;
    nSPEC 		= bsxfun(@minus, SPEC, mu);
    nSPEC 		= bsxfun(@rdivide, nSPEC, stdev);

    rsize 	= buckets.width(find(buckets.width(:)<=size(nSPEC,2),1,'last'));
    rstart  = round((size(nSPEC,2)-rsize)/2);

	inp(:,:) = gpuArray(single(nSPEC(:,rstart:rstart+rsize-1)));

end 

