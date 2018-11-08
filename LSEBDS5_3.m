clear all;
close all;
clc
%%prapare the input/output training
load P5train5inp.mat
x1= veri(:,3); % first input for training
x2= veri (:,2); % second input for training
x3= veri (:,1); % third input fpr training
x=[x1 x2 x3]; % inputs for training
y= veri (:,4); %output for training
%% prapare the input output for testing
load P5test5inp.mat
x1t= veri(:,3); %first input for testing
x2t=veri(:,2); %second input for testing
x3t= veri(:,1); %thrd input for testing
xt= [x1t x2t x3t]; %inputs for testing
yt= veri(:,4); %output for testing
%% ploting
subplot(2,2,1); plot(x1); xlabel('k'); ylabel ('y(k)'); title ('for traininng');
subplot(2,2,3); plot(y); xlabel('k'); ylabel ('y(k)');

subplot(2,2,2); plot(x1t); xlabel('k'); ylabel ('y(k)'); title (' for testing');
subplot(2,2,4); plot(yt); xlabel('k'); ylabel ('y(k)');

teta=[]; %% parameter array/matrix
Xe= [x ones(size(x1))]; %%% extended input

%% membership funtions deffination
%for the first input ,u(k)
x1_array= -2:0.1:2;
    A1 = gaussmf(x1_array, [0.6 -0.2]);
    A2 = gaussmf(x1_array, [0.6 0]);
    A3 = gaussmf(x1_array, [0.8 0]);
    figure(2)
   subplot(3,1,1); plot(x1_array, A1, x1_array, A2,x1_array, A3); 
   xlabel('u(k)');
   title('MFs for u(k) input');

%for the second input ,y(k)
x2_array= -3:0.1:3;
    B1 = gaussmf(x2_array, [0.6 -2]);
    B2 = gaussmf(x2_array, [0.6 2]);
    B3 = gaussmf(x2_array, [0.8 0]);
   subplot(3,1,2); plot(x2_array, B1, x2_array, B2,x2_array, B3);
   xlabel('y(k)');
   title('MFs for y(k) input');
   
%for the third input ,y(k-1)
x3_array= -3:0.1:3;
    C1 = gaussmf(x3_array, [0.6 -2]);
    C2 = gaussmf(x3_array, [0.6 2]);
    C3 = gaussmf(x3_array, [0.8 0]);
   subplot(3,1,3); plot(x3_array, C1, x3_array, C2,x3_array, C3);
   xlabel('y(k-1)');
   title('MFs for y(k-1) input');
%% TRAINING TRANSACTIONS
% %% computing gamma values 
for n= 1:length(x1)
    %finding firing strength (activity degree) of each rule
    w1=gaussmf(x1(n), [0.5 -1.5])*gaussmf(x2(n), [0.6 -2])*gaussmf(x3(n), [0.6 -2]);%A1 B1 C1
    w2=gaussmf(x1(n), [0.5 -1.5])*gaussmf(x2(n), [0.6 -2])*gaussmf(x3(n), [0.6 2]); %A1 B1 C2
    w3=gaussmf(x1(n), [0.5 -1.5])*gaussmf(x2(n), [0.6 -2])*gaussmf(x3(n), [0.8 0]); %A1 B1 C3
    w4=gaussmf(x1(n), [0.5 -1.5])*gaussmf(x2(n), [0.6 2])*gaussmf(x3(n), [0.6 -2]); %A1 B2 C1
    w5=gaussmf(x1(n), [0.5 -1.5])*gaussmf(x2(n), [0.6 2])*gaussmf(x3(n), [0.6 2]); %A1 B2 C2
    w6=gaussmf(x1(n), [0.5 -1.5])*gaussmf(x2(n), [0.6 2])*gaussmf(x3(n), [0.8 0]); %A1 B2 C3
    w7=gaussmf(x1(n), [0.5 -1.5])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.6 -2]);%A1 B3 C1
    w8=gaussmf(x1(n), [0.5 -1.5])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.6 2]); %A1 B3 C2
    w9=gaussmf(x1(n), [0.5 -1.5])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.8 0]);%A1 B3 C3
    w10=gaussmf(x1(n), [0.5 1.5])*gaussmf(x2(n), [0.6 -2])*gaussmf(x3(n), [0.6 -2]); %A2 B1 C1
    w11=gaussmf(x1(n), [0.5 1.5])*gaussmf(x2(n), [0.6 -2])*gaussmf(x3(n), [0.6 2]); %A2 B1 C2
    w12=gaussmf(x1(n), [0.5 1.5])*gaussmf(x2(n), [0.6 -2])*gaussmf(x3(n), [0.8 0]); %A2 B1 C3
    w13=gaussmf(x1(n), [0.5 1.5])*gaussmf(x2(n), [0.6 2])*gaussmf(x3(n), [0.6 -2]); %A2 B2 C1
    w14=gaussmf(x1(n), [0.5 1.5])*gaussmf(x2(n), [0.6 2])*gaussmf(x3(n), [0.6 2]); %A2 B2 C2
    w15=gaussmf(x1(n), [0.5 1.5])*gaussmf(x2(n), [0.6 2])*gaussmf(x3(n), [0.8 0]);%A2 B2 C3
    w16=gaussmf(x1(n), [0.5 1.5])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.6 -2]); %A2 B3 C1
    w17=gaussmf(x1(n), [0.5 1.5])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.6 2]);%A2 B3 C2
    w18=gaussmf(x1(n), [0.5 1.5])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.8 0]); %A2 B3 C3
    w19=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.6 -2])*gaussmf(x3(n), [0.6 -2]); %A3 B1 C1
    w20=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.6 -2])*gaussmf(x3(n), [0.6 2]); %A3 B1 C2
    w21=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.6 -2])*gaussmf(x3(n), [0.8 0]); %A3 B1 C3
    w22=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.6 2])*gaussmf(x3(n), [0.6 -2]);%A3 B2 C1
    w23=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.6 2])*gaussmf(x3(n), [0.6 2]); %A3 B2 C2
    w24=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.6 2])*gaussmf(x3(n), [0.8 0]);%A3 B2 C3
    w25=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.6 -2]); %A3 B3 C1
    w26=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.6 2]); %A3 B3 C2
    w27=gaussmf(x1(n), [0.8 0])*gaussmf(x2(n), [0.8 0])*gaussmf(x3(n), [0.8 0]); %A3 B3 C3
   %finding normalized firing strength (activity degree) of each rule
    gamma1(n)= w1/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma2(n)= w2/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma3(n)= w3/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma4(n)= w4/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma5(n)= w5/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma6(n)= w6/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma7(n)= w7/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma8(n)= w8/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma9(n)= w9/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma10(n)= w10/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma11(n)= w11/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma12(n)= w12/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma13(n)= w13/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma14(n)= w14/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma15(n)= w15/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma16(n)= w16/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma17(n)= w17/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma18(n)= w18/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma19(n)= w19/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma20(n)= w20/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma21(n)= w21/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma12(n)= w22/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma23(n)= w23/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma24(n)= w24/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma15(n)= w25/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma26(n)= w26/(w1+w2+w3+w4+w5+w6+w7+w8);
    gamma27(n)= w27/(w1+w2+w3+w4+w5+w6+w7+w8);

 end

Gama1= diag(gamma1); 
Gama2= diag(gamma2); 
Gama3= diag(gamma3); 
Gama4= diag(gamma4); 
Gama5= diag(gamma5); 
Gama6= diag(gamma6); 
Gama7= diag(gamma7); 
Gama8= diag(gamma8);
Gama9= diag(gamma1); 
Gama10= diag(gamma2); 
Gama11= diag(gamma3); 
Gama12= diag(gamma4); 
Gama13= diag(gamma5); 
Gama14= diag(gamma6); 
Gama15= diag(gamma7); 
Gama16= diag(gamma8);
Gama17= diag(gamma1); 
Gama18= diag(gamma2); 
Gama19= diag(gamma3); 
Gama20= diag(gamma4); 
Gama21= diag(gamma5); 
Gama22= diag(gamma6); 
Gama23= diag(gamma7); 
Gama24= diag(gamma8);
Gama25= diag(gamma6); 
Gama26= diag(gamma7); 
Gama27= diag(gamma8);
%%optimal least-squares solution
%Xussu=[Gama1*Xe Gama2*Xe Gama3*Xe Gama4*Xe  Gama5*Xe Gama6*Xe Gama7*Xe ];
%teta1 = inv(Xussu'*Xussu)*Xussu'*y; %its results are NaNbecaouse of zero determinant

%%weighted least-squares approach appliedper rule
teta=[inv(Xe'*Gama1*Xe)*Xe'*Gama1*y inv(Xe'*Gama2*Xe)*Xe'*Gama2*y inv(Xe'*Gama3*Xe)*Xe'*Gama3*y inv(Xe'*Gama4*Xe)*Xe'*Gama4*y...
      inv(Xe'*Gama5*Xe)*Xe'*Gama5*y inv(Xe'*Gama6*Xe)*Xe'*Gama6*y inv(Xe'*Gama7*Xe)*Xe'*Gama7*y inv(Xe'*Gama8*Xe)*Xe'*Gama8*y...
      inv(Xe'*Gama9*Xe)*Xe'*Gama9*y inv(Xe'*Gama10*Xe)*Xe'*Gama10*y inv(Xe'*Gama11*Xe)*Xe'*Gama11*y inv(Xe'*Gama12*Xe)*Xe'*Gama12*y...
      inv(Xe'*Gama13*Xe)*Xe'*Gama13*y inv(Xe'*Gama14*Xe)*Xe'*Gama14*y inv(Xe'*Gama15*Xe)*Xe'*Gama15*y inv(Xe'*Gama16*Xe)*Xe'*Gama16*y...
      inv(Xe'*Gama17*Xe)*Xe'*Gama17*y inv(Xe'*Gama18*Xe)*Xe'*Gama18*y inv(Xe'*Gama19*Xe)*Xe'*Gama19*y inv(Xe'*Gama20*Xe)*Xe'*Gama20*y...
      inv(Xe'*Gama21*Xe)*Xe'*Gama21*y inv(Xe'*Gama22*Xe)*Xe'*Gama22*y inv(Xe'*Gama23*Xe)*Xe'*Gama23*y inv(Xe'*Gama24*Xe)*Xe'*Gama24*y...
      inv(Xe'*Gama25*Xe)*Xe'*Gama25*y inv(Xe'*Gama26*Xe)*Xe'*Gama26*y inv(Xe'*Gama27*Xe)*Xe'*Gama27*y ];
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
              p9= teta(1,9)*x1(i)+teta(2,9)*x2(i)+teta(3,9)*x3(i)+ teta(4,9);
              p10= teta(1,10)*x1(i)+teta(2,10)*x2(i)+teta(3,10)*x3(i)+ teta(4,10);
              p11= teta(1,11)*x1(i)+teta(2,11)*x2(i)+teta(3,11)*x3(i)+ teta(4,11);
              p12= teta(1,12)*x1(i)+teta(2,12)*x2(i)+teta(3,12)*x3(i)+ teta(4,12);
              p13= teta(1,13)*x1(i)+teta(2,13)*x2(i)+teta(3,13)*x3(i)+ teta(4,13);
              p14= teta(1,14)*x1(i)+teta(2,14)*x2(i)+teta(3,14)*x3(i)+ teta(4,14);
              p15= teta(1,15)*x1(i)+teta(2,15)*x2(i)+teta(3,15)*x3(i)+ teta(4,15);
              p16= teta(1,16)*x1(i)+teta(2,16)*x2(i)+teta(3,16)*x3(i)+ teta(4,16);
              p17= teta(1,17)*x1(i)+teta(2,17)*x2(i)+teta(3,17)*x3(i)+ teta(4,17);
              p18= teta(1,18)*x1(i)+teta(2,18)*x2(i)+teta(3,18)*x3(i)+ teta(4,18);
              p19= teta(1,19)*x1(i)+teta(2,19)*x2(i)+teta(3,19)*x3(i)+ teta(4,19);
              p20= teta(1,20)*x1(i)+teta(2,20)*x2(i)+teta(3,20)*x3(i)+ teta(4,20);
              p21= teta(1,21)*x1(i)+teta(2,21)*x2(i)+teta(3,21)*x3(i)+ teta(4,21);
              p22= teta(1,22)*x1(i)+teta(2,22)*x2(i)+teta(3,22)*x3(i)+ teta(4,22);
              p23= teta(1,23)*x1(i)+teta(2,23)*x2(i)+teta(3,23)*x3(i)+ teta(4,23);
              p24= teta(1,24)*x1(i)+teta(2,24)*x2(i)+teta(3,24)*x3(i)+ teta(4,24);
              p25= teta(1,25)*x1(i)+teta(2,25)*x2(i)+teta(3,25)*x3(i)+ teta(4,25);
              p26= teta(1,26)*x1(i)+teta(2,26)*x2(i)+teta(3,26)*x3(i)+ teta(4,26);
              p27= teta(1,27)*x1(i)+teta(2,27)*x2(i)+teta(3,27)*x3(i)+ teta(4,27);
    w1=gaussmf(x1(i), [0.5 -1.5])*gaussmf(x2(i), [0.6 -2])*gaussmf(x3(i), [0.6 -2]);%A1 B1 C1
    w2=gaussmf(x1(i), [0.5 -1.5])*gaussmf(x2(i), [0.6 -2])*gaussmf(x3(i), [0.6 2]); %A1 B1 C2
    w3=gaussmf(x1(i), [0.5 -1.5])*gaussmf(x2(i), [0.6 -2])*gaussmf(x3(i), [0.8 0]); %A1 B1 C3
    w4=gaussmf(x1(i), [0.5 -1.5])*gaussmf(x2(i), [0.6 2])*gaussmf(x3(i), [0.6 -2]); %A1 B2 C1
    w5=gaussmf(x1(i), [0.5 -1.5])*gaussmf(x2(i), [0.6 2])*gaussmf(x3(i), [0.6 2]); %A1 B2 C2
    w6=gaussmf(x1(i), [0.5 -1.5])*gaussmf(x2(i), [0.6 2])*gaussmf(x3(i), [0.8 0]); %A1 B2 C3
    w7=gaussmf(x1(i), [0.5 -1.5])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.6 -2]);%A1 B3 C1
    w8=gaussmf(x1(i), [0.5 -1.5])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.6 2]); %A1 B3 C2
    w9=gaussmf(x1(i), [0.5 -1.5])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.8 0]);%A1 B3 C3
    w10=gaussmf(x1(i), [0.5 1.5])*gaussmf(x2(i), [0.6 -2])*gaussmf(x3(i), [0.6 -2]); %A2 B1 C1
    w11=gaussmf(x1(i), [0.5 1.5])*gaussmf(x2(i), [0.6 -2])*gaussmf(x3(i), [0.6 2]); %A2 B1 C2
    w12=gaussmf(x1(i), [0.5 1.5])*gaussmf(x2(i), [0.6 -2])*gaussmf(x3(i), [0.8 0]); %A2 B1 C3
    w13=gaussmf(x1(i), [0.5 1.5])*gaussmf(x2(i), [0.6 2])*gaussmf(x3(i), [0.6 -2]); %A2 B2 C1
    w14=gaussmf(x1(i), [0.5 1.5])*gaussmf(x2(i), [0.6 2])*gaussmf(x3(i), [0.6 2]); %A2 B2 C2
    w15=gaussmf(x1(i), [0.5 1.5])*gaussmf(x2(i), [0.6 2])*gaussmf(x3(i), [0.8 0]);%A2 B2 C3
    w16=gaussmf(x1(i), [0.5 1.5])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.6 -2]); %A2 B3 C1
    w17=gaussmf(x1(i), [0.5 1.5])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.6 2]);%A2 B3 C2
    w18=gaussmf(x1(i), [0.5 1.5])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.8 0]); %A2 B3 C3
    w19=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.6 -2])*gaussmf(x3(i), [0.6 -2]); %A3 B1 C1
    w20=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.6 -2])*gaussmf(x3(i), [0.6 2]); %A3 B1 C2
    w21=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.6 -2])*gaussmf(x3(i), [0.8 0]); %A3 B1 C3
    w22=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.6 2])*gaussmf(x3(i), [0.6 -2]);%A3 B2 C1
    w23=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.6 2])*gaussmf(x3(i), [0.6 2]); %A3 B2 C2
    w24=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.6 2])*gaussmf(x3(i), [0.8 0]);%A3 B2 C3
    w25=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.6 -2]); %A3 B3 C1
    w26=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.6 2]); %A3 B3 C2
    w27=gaussmf(x1(i), [0.8 0])*gaussmf(x2(i), [0.8 0])*gaussmf(x3(i), [0.8 0]); %A3 B3 C3
       train_result(i)=(w1*p1+w2*p2+w3*p3+w4*p4+w5*p5+w6*p6+w7*p7+w8*p8+w9*p9...
                        +w10*p10+w11*p11+w12*p12+w13*p13+w14*p14+w15*p15+w16*p16+w17*p17+w18*p18...
                        +w19*p19+w20*p20+w21*p21+w22*p22+w23*p23+w24*p24+w25*p25+w26*p26+w27*p27)/...
                        (w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12+w13+w14+w15+w16+w17+w18+w19+w20+w21+w22+w23+w24+w25+w26+w27);
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
              p9= teta(1,9)*x1t(i)+teta(2,9)*x2t(i)+teta(3,9)*x3t(i)+ teta(4,9);
              p10= teta(1,10)*x1t(i)+teta(2,10)*x2t(i)+teta(3,10)*x3t(i)+ teta(4,10);
              p11= teta(1,11)*x1t(i)+teta(2,11)*x2t(i)+teta(3,11)*x3t(i)+ teta(4,11);
              p12= teta(1,12)*x1t(i)+teta(2,12)*x2t(i)+teta(3,12)*x3t(i)+ teta(4,12);
              p13= teta(1,13)*x1t(i)+teta(2,13)*x2t(i)+teta(3,13)*x3t(i)+ teta(4,13);
              p14= teta(1,14)*x1t(i)+teta(2,14)*x2t(i)+teta(3,14)*x3t(i)+ teta(4,14);
              p15= teta(1,15)*x1t(i)+teta(2,15)*x2t(i)+teta(3,15)*x3t(i)+ teta(4,15);
              p16= teta(1,16)*x1t(i)+teta(2,16)*x2t(i)+teta(3,16)*x3t(i)+ teta(4,16);
              p17= teta(1,17)*x1t(i)+teta(2,17)*x2t(i)+teta(3,17)*x3t(i)+ teta(4,17);
              p18= teta(1,18)*x1t(i)+teta(2,18)*x2t(i)+teta(3,18)*x3t(i)+ teta(4,18);
              p19= teta(1,19)*x1t(i)+teta(2,19)*x2t(i)+teta(3,19)*x3t(i)+ teta(4,19);
              p20= teta(1,20)*x1t(i)+teta(2,20)*x2t(i)+teta(3,20)*x3t(i)+ teta(4,20);
              p21= teta(1,21)*x1t(i)+teta(2,21)*x2t(i)+teta(3,21)*x3t(i)+ teta(4,21);
              p22= teta(1,22)*x1t(i)+teta(2,22)*x2t(i)+teta(3,22)*x3t(i)+ teta(4,22);
              p23= teta(1,23)*x1t(i)+teta(2,23)*x2t(i)+teta(3,23)*x3t(i)+ teta(4,23);
              p24= teta(1,24)*x1t(i)+teta(2,24)*x2t(i)+teta(3,24)*x3t(i)+ teta(4,24);
              p25= teta(1,25)*x1t(i)+teta(2,25)*x2t(i)+teta(3,25)*x3t(i)+ teta(4,25);
              p26= teta(1,26)*x1t(i)+teta(2,26)*x2t(i)+teta(3,26)*x3t(i)+ teta(4,26);
              p27= teta(1,27)*x1t(i)+teta(2,27)*x2t(i)+teta(3,27)*x3t(i)+ teta(4,27);
   w1=gaussmf(x1t(i), [0.5 -1.5])*gaussmf(x2t(i), [0.6 -2])*gaussmf(x3t(i), [0.6 -2]);%A1 B1 C1
    w2=gaussmf(x1t(i), [0.5 -1.5])*gaussmf(x2t(i), [0.6 -2])*gaussmf(x3t(i), [0.6 2]); %A1 B1 C2
    w3=gaussmf(x1t(i), [0.5 -1.5])*gaussmf(x2t(i), [0.6 -2])*gaussmf(x3t(i), [0.8 0]); %A1 B1 C3
    w4=gaussmf(x1t(i), [0.5 -1.5])*gaussmf(x2t(i), [0.6 2])*gaussmf(x3t(i), [0.6 -2]); %A1 B2 C1
    w5=gaussmf(x1t(i), [0.5 -1.5])*gaussmf(x2t(i), [0.6 2])*gaussmf(x3t(i), [0.6 2]); %A1 B2 C2
    w6=gaussmf(x1t(i), [0.5 -1.5])*gaussmf(x2t(i), [0.6 2])*gaussmf(x3t(i), [0.8 0]); %A1 B2 C3
    w7=gaussmf(x1t(i), [0.5 -1.5])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.6 -2]);%A1 B3 C1
    w8=gaussmf(x1t(i), [0.5 -1.5])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.6 2]); %A1 B3 C2
    w9=gaussmf(x1t(i), [0.5 -1.5])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.8 0]);%A1 B3 C3
    w10=gaussmf(x1t(i), [0.5 1.5])*gaussmf(x2t(i), [0.6 -2])*gaussmf(x3t(i), [0.6 -2]); %A2 B1 C1
    w11=gaussmf(x1t(i), [0.5 1.5])*gaussmf(x2t(i), [0.6 -2])*gaussmf(x3t(i), [0.6 2]); %A2 B1 C2
    w12=gaussmf(x1t(i), [0.5 1.5])*gaussmf(x2t(i), [0.6 -2])*gaussmf(x3t(i), [0.8 0]); %A2 B1 C3
    w13=gaussmf(x1t(i), [0.5 1.5])*gaussmf(x2t(i), [0.6 2])*gaussmf(x3t(i), [0.6 -2]); %A2 B2 C1
    w14=gaussmf(x1t(i), [0.5 1.5])*gaussmf(x2t(i), [0.6 2])*gaussmf(x3t(i), [0.6 2]); %A2 B2 C2
    w15=gaussmf(x1t(i), [0.5 1.5])*gaussmf(x2t(i), [0.6 2])*gaussmf(x3t(i), [0.8 0]);%A2 B2 C3
    w16=gaussmf(x1t(i), [0.5 1.5])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.6 -2]); %A2 B3 C1
    w17=gaussmf(x1t(i), [0.5 1.5])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.6 2]);%A2 B3 C2
    w18=gaussmf(x1t(i), [0.5 1.5])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.8 0]); %A2 B3 C3
    w19=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.6 -2])*gaussmf(x3t(i), [0.6 -2]); %A3 B1 C1
    w20=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.6 -2])*gaussmf(x3t(i), [0.6 2]); %A3 B1 C2
    w21=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.6 -2])*gaussmf(x3t(i), [0.8 0]); %A3 B1 C3
    w22=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.6 2])*gaussmf(x3t(i), [0.6 -2]);%A3 B2 C1
    w23=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.6 2])*gaussmf(x3t(i), [0.6 2]); %A3 B2 C2
    w24=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.6 2])*gaussmf(x3t(i), [0.8 0]);%A3 B2 C3
    w25=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.6 -2]); %A3 B3 C1
    w26=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.6 2]); %A3 B3 C2
    w27=gaussmf(x1t(i), [0.8 0])*gaussmf(x2t(i), [0.8 0])*gaussmf(x3t(i), [0.8 0]); %A3 B3 C3
       test_result(i)=(w1*p1+ w2*p2+w3*p3+w4*p4+w5*p5+w6*p6+w7*p7+w8*p8+w9*p9+w10*p10+w11*p11+w12*p12+ w13*p13+w14*p14+w15*p15+w16*p16+w17*p17+w18*p18*w19*p19+w20*p20+w21*p21+w22*p22+w23*p23+w24*p24+w25*p25+w26*p26+w27*p27)/(w1+w2+w3+w4+w5+w6+w7+w8+w9+w10+w11+w12+w13+w14+w15+w16+w17+w18+w19+w20+w21+w22+w23+w24+w25+w26+w27);
    end


    figure(4);
    plot(test_result,'r-.'),
    title('testing performance of fuzzy system');
    hold on
    plot(yt,'k'); hold off
    xlabel('k'), ylabel('y(k)'); legend('obtained', 'desired')
%% END of PERFORMANCE FOR TESTING DATA with founded rule parameter



