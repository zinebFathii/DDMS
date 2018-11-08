clear all;
close all;
clc
%%prapare the input/output training
load P4veri.mat
x1= veri(:,3); % first input for training
x2= veri (:,2); % second input for training
x3= veri (:,1); % third input fpr training
x=[x1 x2 x3]; % inputs for training
y= veri (:,4); %output for training
%% prapare the input output for testing
load P4veritest.mat
x1t= veri(:,3); %first input for testing
x2t=veri(:,2); %second input for testing
x3t= veri(:,1); %thrd input for testing
xt= [x1t x2t x3t]; %inputs for testing
yt= veri(:,4); %output for testing
%% ploting
subplot(2,2,1); plot(x1); xlabel('k'); ylabel ('u(k)'); title ('for traininng');
subplot(2,2,3); plot(y); xlabel('k'); ylabel ('y(k)');

subplot(2,2,2); plot(x1t); xlabel('k'); ylabel ('u(k)'); title (' for testing');
subplot(2,2,4); plot(yt); xlabel('k'); ylabel ('y(k)');

teta=[]; %% parameter array/matrix
Xe= [x ones(size(x1))]; %%% extended input

%% membership funtions deffination
%for the first input ,u(k)
x1_array= -1:0.1:4;
    A1 = gaussmf(x1_array, [0.8 0]);
    A2 = gaussmf(x1_array, [1 2.5]);
    figure(2)
   subplot(3,1,1); plot(x1_array, A1, x1_array, A2); 
   xlabel('u(k)');
   title('MFs for u(k) input');

%for the second input ,y(k)
x2_array= -1:0.1:4;
    B1 = gaussmf(x2_array, [0.8 0]);
    B2 = gaussmf(x2_array, [1  2.5]);
   subplot(3,1,2); plot(x2_array, B1, x2_array, B2);
   xlabel('y(k)');
   title('MFs for y(k) input');
   
%for the third input ,y(k-1)
x3_array= -1:0.1:4;
    C1 = gaussmf(x3_array, [0.8 0]);
    C2 = gaussmf(x3_array, [1 2.5]);
   subplot(3,1,3); plot(x3_array, C1, x3_array, C2);
   xlabel('y(k-1)');
   title('MFs for y(k-1) input');
%% TRAINING TRANSACTIONS
% %% computing gamma values 
for n= 1:length(x1)
    %finding firing strength (activity degree) of each rule
    w1=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.8 0]);%A1 B1 C1
    w2=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [1 2.5]); %A1 B1 C2
    w3=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [1 2.5])*gaussmf(x3(n), [0.8 0]); %A1 B2 C1
    w4=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [1 2.5])*gaussmf(x3(n), [1 2.5]); %A1 B2 C2
    w5=gaussmf(x1(n), [1 2.5])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.8 0]); %A2 B1 C1
    w6=gaussmf(x1(n), [1 2.5])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [1 2.5]); %A2 B1 C2
    w7=gaussmf(x1(n), [1 2.5])*gaussmf(x2(n), [1 2.5])*gaussmf(x3(n), [0.8 0]);%A2 B2 C1
    w8=gaussmf(x1(n), [1 2.5])*gaussmf(x2(n), [1 2.5])*gaussmf(x3(n), [1 2.5]); %A2 B2 C2
   %finding normalized firing strength (activity degree) of each rule
    gamma1(n)= w1/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma2(n)= w2/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma3(n)= w3/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma4(n)= w4/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma5(n)= w5/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma6(n)= w6/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma7(n)= w7/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma8(n)= w8/(w1+w2+w3+w4+w5+w6+w7+w8);

 end

Gama1= diag(gamma1); 
Gama2= diag(gamma2); 
Gama3= diag(gamma3); 
Gama4= diag(gamma4); 
Gama5= diag(gamma5); 
Gama6= diag(gamma6); 
Gama7= diag(gamma7); 
Gama8= diag(gamma8);
%%optimal least-squares solution
%Xussu=[Gama1*Xe Gama2*Xe Gama3*Xe Gama4*Xe  Gama5*Xe Gama6*Xe Gama7*Xe ];
%teta1 = inv(Xussu'*Xussu)*Xussu'*y; %its results are NaNbecaouse of zero determinant

%%weighted least-squares approach appliedper rule
teta=[inv(Xe'*Gama1*Xe)*Xe'*Gama1*y inv(Xe'*Gama2*Xe)*Xe'*Gama2*y inv(Xe'*Gama3*Xe)*Xe'*Gama3*y inv(Xe'*Gama4*Xe)*Xe'*Gama4*y...
      inv(Xe'*Gama5*Xe)*Xe'*Gama5*y inv(Xe'*Gama6*Xe)*Xe'*Gama6*y inv(Xe'*Gama7*Xe)*Xe'*Gama7*y inv(Xe'*Gama8*Xe)*Xe'*Gama8*y];

  %% END OF TRAINING

%% PERFORMANCE FOR TRAINING DATA with founded rule parameter 

    for i= 1:length(x1)


              p1= teta(1,1)*x1(i)+teta(2,1)*x2(i)+teta(3,1)*x3(i)+ teta(4,1);
              p2= teta(1,2)*x1(i)+teta(2,2)*x2(i)+teta(3,2)*x3(i)+ teta(4,2);
              p3= teta(1,3)*x1(i)+teta(2,3)*x2(i)+teta(3,3)*x3(i)+ teta(4,3);
              p4= teta(1,4)*x1(i)+teta(2,4)*x2(i)+teta(3,4)*x3(i)+ teta(4,4);
              p5= teta(1,5)*x1(i)+teta(2,5)*x2(i)+teta(3,5)*x3(i)+ teta(4,5);
              p6= teta(1,6)*x1(i)+teta(2,6)*x2(i)+teta(3,6)*x3(i)+ teta(4,6);
              p7= teta(1,7)*x1(i)+teta(2,7)*x2(i)+teta(3,7)*x3(i)+ teta(4,7);
              p8= teta(1,8)*x1(i)+teta(2,8)*x2(i)+teta(3,8)*x3(i)+ teta(4,8);
        w1=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.8 0]);%A1 B1 C1
        w2=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [1 2.5]); %A1 B1 C2
        w3=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [1 2.5])*gaussmf(x3(i), [0.8 0]); %A1 B2 C1
        w4=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [1 2.5])*gaussmf(x3(i), [1 2.5]); %A1 B2 C2
        w5=gaussmf(x1(i), [1 2.5])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.8 0]); %A2 B1 C1
        w6=gaussmf(x1(i), [1 2.5])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [1 2.5]); %A2 B1 C2
        w7=gaussmf(x1(i), [1 2.5])*gaussmf(x2(i), [1 2.5])*gaussmf(x3(i), [0.8 0]);%A2 B2 C1
        w8=gaussmf(x1(i), [1 2.5])*gaussmf(x2(i), [1 2.5])*gaussmf(x3(i), [1 2.5]); %A2 B2 C2
 
       train_result(i)=(w1*p1+ w2*p2+w3*p3+w4*p4+w5*p5+w6*p6+w7*p7+w8*p8)/(w1+w2+w3+w4+w5+w6+w7+w8);
    end


    figure(3);
    plot(train_result,'g.'),
    title('training performance of fuzzy system');
    hold on
    plot(y,'k'); hold off
    xlabel('k'), ylabel('y(k)'); legend('obtained', 'desired')


%% END of PERFORMANCE FOR TRAINING DATA with founded rule parameter
% TESTING TRANSACTIONS
%% PERFORMANCE FOR TESTING DATA with founded rule parameter 

    for i= 1:length(x1t)

              p1= teta(1,1)*x1t(i)+teta(2,1)*x2t(i)+teta(3,1)*x3t(i)+ teta(4,1);
              p2= teta(1,2)*x1t(i)+teta(2,2)*x2t(i)+teta(3,2)*x3t(i)+ teta(4,2);
              p3= teta(1,3)*x1t(i)+teta(2,3)*x2t(i)+teta(3,3)*x3t(i)+ teta(4,3);
              p4= teta(1,4)*x1t(i)+teta(2,4)*x2t(i)+teta(3,4)*x3t(i)+ teta(4,4);
              p5= teta(1,5)*x1t(i)+teta(2,5)*x2t(i)+teta(3,5)*x3t(i)+ teta(4,5);
              p6= teta(1,6)*x1t(i)+teta(2,6)*x2t(i)+teta(3,6)*x3t(i)+ teta(4,6);
              p7= teta(1,7)*x1t(i)+teta(2,7)*x2t(i)+teta(3,7)*x3t(i)+ teta(4,7);
              p8= teta(1,8)*x1t(i)+teta(2,8)*x2t(i)+teta(3,8)*x3t(i)+ teta(4,8);
        w1=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.8 0]);%A1 B1 C1
        w2=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [1 2.5]); %A1 B1 C2
        w3=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [1 2.5])*gaussmf(x3t(i), [0.8 0]); %A1 B2 C1
        w4=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [1 2.5])*gaussmf(x3t(i), [1 2.5]); %A1 B2 C2
        w5=gaussmf(x1t(i), [1 2.5])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.8 0]); %A2 B1 C1
        w6=gaussmf(x1t(i), [1 2.5])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [1 2.5]); %A2 B1 C2
        w7=gaussmf(x1t(i), [1 2.5])*gaussmf(x2t(i), [1 2.5])*gaussmf(x3t(i), [0.8 0]);%A2 B2 C1
        w8=gaussmf(x1t(i), [1 2.5])*gaussmf(x2t(i), [1 2.5])*gaussmf(x3t(i), [1 2.5]); %A2 B2 C2
 
       test_result(i)=(w1*p1+ w2*p2+w3*p3+w4*p4+w5*p5+w6*p6+w7*p7+w8*p8)/(w1+w2+w3+w4+w5+w6+w7+w8);
    end


    figure(4);
    plot(test_result,'r-.'),
    title('testing performance of fuzzy system');
    hold on
    plot(yt,'k'); hold off
    xlabel('k'), ylabel('y(k)'); legend('obtained', 'desired')
%% END of PERFORMANCE FOR TESTING DATA with founded rule parameter



