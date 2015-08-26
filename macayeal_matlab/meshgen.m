% This script creates the mesh for the EISMINT Level 1
% ice shelf test.
nodes=17*21;
imax=21;
jmax=17;
nrows=17*21*2;
row=zeros(nrows,1);
col=zeros(nrows,1);
value=zeros(nrows,1);
xy=zeros(nrows,2);
gamma=zeros(imax,jmax);
Boundu=zeros(53,1); % zero x-velocity nodes
Boundv=zeros(37,1); % zero y-velocity nodes
count=0;
dx=100e3/L; % L is the nondimensional length scale
dy=80e3/L;
for i=1:imax
for j=1:jmax
count=count+1;
xy(count,1)=(i-1)/(imax-1)*dx;
xy(count,2)=(j-1)/(jmax-1)*dy;
gamma(i,j)=count;
end
end
% Create triangulation
%
nel=(imax-1)*(jmax-1)*2;
index=zeros(nel,3);
count=0;
for i=1:imax-1
for j=1:jmax-1
count=count+1;
index(count,1)=gamma(i,j);
index(count,2)=gamma(i+1,j);
index(count,3)=gamma(i+1,j+1);
count=count+1;
index(count,1)=gamma(i,j);
index(count,2)=gamma(i+1,j+1);
index(count,3)=gamma(i,j+1);
end
end
bxcount=0;
bycount=0;
for j=1:(jmax-1) % left side
bxcount=bxcount+1;
Boundu(bxcount)=gamma(1,j);
bycount=bycount+1;
Boundv(bycount)=gamma(1,j);
end
for i=1:imax % top side
bxcount=bxcount+1;
bycount=bycount+1;
Boundu(bxcount)=gamma(i,jmax);
Boundv(bycount)=gamma(i,jmax);
end
for j=1:(jmax-1) % right side
bxcount=bxcount+1;
Boundu(bxcount)=gamma(imax,j);
end
% Create adjacency matrix and plot mesh and silhouette of mesh
count=0;
pause
for n=1:nel
for i=1:3
for j=1:3
count=count+1;
row(count)=index(n,i)*2-1;
col(count)=index(n,j)*2-1;
value(count)=1;
count=count+1;
row(count)=index(n,i)*2-1;
col(count)=index(n,j)*2;
value(count)=1;
count=count+1;
row(count)=index(n,i)*2;
col(count)=index(n,j)*2-1;
value(count)=1;
row(count)=index(n,i)*2;
col(count)=index(n,j)*2;
value(count)=1;
end
end
end
% Construct mesh plot
Adj=sparse(row,col,value);
gplot(Adj,xy);
pause
spy(Adj)
