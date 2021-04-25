%%
function u = gsprocess(v,N,n)
	u = zeros(N,n);
	for k=1:n
		if k == 1
			u(:,k) = v(:,k); u(:,k) = u(:,k)/norm(u(:,k));
		else
			A = zeros(N,1);
			for l=1:k-1
				A = A + (v(:,k)'*u(:,l))*u(:,l);
			end
			u(:,k) = v(:,k) - A;
			u(:,k) = u(:,k)/norm(u(:,k));
		end
	end
end