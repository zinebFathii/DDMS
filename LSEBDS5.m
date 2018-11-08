clear all;
close all;
clc
%% prepare the inputs/output for training
load P5veri.mat

x1= veri(:,2); % first input for training
x2= veri(:,1); %second input for trainig
x=[x1 x2];
y= veri(:,3); %output for training
%% prepare the input/output for testing
load P5veritest.mat
x1t= veri (:,2); %first input for tetig
x2t= veri (:,1); %second input for testing
xt=[x1t x2t]; % input for testig
yt= veri(:,3); %% output for testing
%%% ploting
subplot(2,2,1); plot(x1); xlabel('uk'); ylabel('y(k)'); title('for training');
subplot (2,2,3); plot(y); xlabel('uk'); ylabel ('y(k)');

subplot(2,2,2); plot(x1t); xlabel('uk'); ylabel('y(k)'); title('for training');
subplot (2,2,4); plot(yt); xlabel('uk'); ylabel ('y(k)');
 
teta=[]; %% parameter array/matrix
Xe= [x ones(size(x1))]; %% extended input
%% membership funtions defination 
%% for the first input u(k)
x1_array= -2:0.1:2;
 A1= gaussmf(x1_array, [0.6 -0.2]);
 A2= gaussmf(x1_array, [0.6 0]);
 figure (2)
 subplot(2,1,1); plot(x1_array, A1, x1_array, A2);
 xlabel('u(k)');
 title ('MFs for u(k) input');
 
 x2_array= -3:0.1:4;
 B1= gaussmf(x2_array, [0.8 0]);
 B2= gaussmf(x2_array, [0.8 1]);
 subplot (2,1,2); plot (x2_array, B1, x2_array,B2);
 xlabel('y(k)');
 title ('MFs for y(k) input');
 % TRAINING TRANSACTIONS
% %% computing gamma values 
for n= 1:length(x1)
    %finding firing strength (activity degree) of each rule
    w1=gaussmf(x1(n), [0.6 -0.2])*gaussmf(x2(n), [0.8 0]);%A1 B1 
    w2=gaussmf(x1(n), [0.6 -0.2])*gaussmf(x2(n), [0.8 0]); %A1 B2 
    w3=gaussmf(x1(n), [0.6 0])*gaussmf(x2(n), [0.8 1]); %A2 B1 
    w4=gaussmf(x1(n), [0.6 0])*gaussmf(x2(n), [0.8 1]); %A2 B2 
   
   %finding normalized firing strength (activity degree) of each rule
    gamma1(n)= w1/(w1+w2+w3+w4);
    gamma2(n)= w2/(w1+w2+w3+w4);
    gamma3(n)= w3/(w1+w2+w3+w4);
    gamma4(n)= w4/(w1+w2+w3+w4);
   
 end

Gama1= diag(gamma1); 
Gama2= diag(gamma2); 
Gama3= diag(gamma3); 
Gama4= diag(gamma4); 


%%weighted least-squares approach appliedper rule
teta=[inv(Xe'*Gama1*Xe)*Xe'*Gama1*y inv(Xe'*Gama2*Xe)*Xe'*Gama2*y inv(Xe'*Gama3*Xe)*Xe'*Gama3*y inv(Xe'*Gama4*Xe)*Xe'*Gama4*y];

  %% END OF TRAINING

%% PERFORMANCE FOR TRAINING DATA with founded rule parameter 

    for i= 1:length(x1)


              p1= teta(1,1)*x1(i)+teta(2,1)*x2(i);
              p2= teta(1,2)*x1(i)+teta(2,2)*x2(i);
              p3= teta(1,3)*x1(i)+teta(2,3)*x2(i);
              p4= teta(1,4)*x1(i)+teta(2,4)*x2(i);
              
        w1=gaussmf(x1(i), [0.6 -0.2])*gaussmf(x2(i), [0.8 0]);%A1 B1 
        w2=gaussmf(x1(i), [0.6 -0.2])*gaussmf(x2(i), [0.8 0]); %A1 B2 
        w3=gaussmf(x1(i), [0.6 0])*gaussmf(x2(i), [0.8 1]); %A2 B1 
        w4=gaussmf(x1(i), [0.6 0])*gaussmf(x2(i), [0.8 1]); %A2 B2 
        
 
       train_result(i)=(w1*p1+ w2*p2+w3*p3+w4*p4)/(w1+w2+w3+w4);
    end


    figure(3);
    plot(train_result,'b-.'),
    title('training performance of fuzzy system');
    hold on
    plot(y,'k'); hold off
    xlabel('k'), ylabel('y(k)'); legend('obtained', 'desired')


%% END of PERFORMANCE FOR TRAINING DATA with founded rule parameter
% TESTING TRANSACTIONS
%% PERFORMANCE FOR TESTING DATA with founded rule parameter 

    for i= 1:length(x1t)

              p1= teta(1,1)*x1t(i)+teta(2,1)*x2t(i);
              p2= teta(1,2)*x1t(i)+teta(2,2)*x2t(i);
              p3= teta(1,3)*x1t(i)+teta(2,3)*x2t(i);
              p4= teta(1,4)*x1t(i)+teta(2,4)*x2t(i);
             
        w1=gaussmf(x1t(i), [0.6 -0.2])*gaussmf(x2t(i), [0.8 0]);%A1 B1 
        w2=gaussmf(x1t(i), [0.6 -0.2])*gaussmf(x2t(i), [0.8 0]); %A1 B2 
        w3=gaussmf(x1t(i), [0.6 0])*gaussmf(x2t(i), [0.8 1]); %A2 B1 
        w4=gaussmf(x1t(i), [0.6 0])*gaussmf(x2t(i), [0.8 1]); %A2 B2 
        
 
       test_result(i)=(w1*p1+ w2*p2+w3*p3+w4*p4)/(w1+w2+w3+w4);
    end


    figure(4);
    plot(test_result,'r-*'),
    title('testing performance of fuzzy system');
    hold on
    plot(yt,'k'); hold off
    xlabel('k'), ylabel('y(k)'); legend('obtained', 'desired')
%% END of PERFORMANCE FOR TESTING DATA with founded rule parameter


 