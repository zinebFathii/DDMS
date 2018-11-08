% AYRIK DERECEDEN Y�KSEK DO�RUSAL OLMAYAN D�NAM�K S�STEM-Problem_1
% veri k�mesi haz�rlan�r.
% problem1
function veri=P1datahazirla(deger)

dataX=zeros(100,4);

for k=1:100
    if deger==0
        dataX(k,3)=cos(2*pi*k/100); % e�itim seti i�in 
    elseif deger==1
         dataX(k,3)=sin(2*pi*k/25); % test seti i�in 
    end
                     
          if k==1 
            dataX(k,1)=0;
            dataX(k,2)=0;
             dataX(k,4)=(((0*0*(0+2.5)))/(1+0)^2+0^2)+dataX(k,3); %y(k)
          elseif k==2
            dataX(k,1)=dataX(k-1,4);
            dataX(k,2)=0;
            dataX(k,4)=(((dataX(k-1,4)*0*(dataX(k-1,4)+2.5)))/(1+dataX(k-1,4)^2+0^2))+dataX(k,3); %y(k)
        else
            dataX(k,1)=dataX(k-1,4);
            dataX(k,2)=dataX(k-2,4);
            dataX(k,4)=(((dataX(k-1,4)*dataX(k-2,4)*(dataX(k-1,4)+2.5)))/(1+dataX(k-1,4)^2+dataX(k-2,4)^2))+dataX(k,3); %y(k)
        end

       
end

for k=1:100
    veri(k,1)=dataX(k,1);%k-1
    veri(k,2)=dataX(k,2);%k-2
    veri(k,3)=dataX(k,3);%u(k)
    veri(k,4)=dataX(k,4);%y(k)
end

 %%
%PLOTING
 if deger==0
     
     figure(1);
     plot(veri(:,4)), xlabel(''), ylabel('y(k)');
     title('E�itim seti i�in Veri K�mesi');
 
     figure(2);
     plot(veri(:,3)), xlabel(''), ylabel('u(k)');
     title('E�itim Seti i�in u(k)');
     save veri
 elseif deger==1

     figure(3);
     plot(veri(:,4)), xlabel(''), ylabel('y(k)');
     title('Test seti i�in Veri K�mesi');
   
     figure(4);
     plot(veri(:,3)), xlabel(''), ylabel('u(k)');
     title('Test seti i�in u(k)');
     save veritest
     
 end

end