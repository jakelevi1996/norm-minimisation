% This is some horrible code, but I really don't see the point in trying to
% understand how to use MATLAB structs and cell arrays given that they are
% simply abhorrent

disp('Converting b vectors')

load b1.mat
csvwrite('b1.csv', b1);

load b2.mat
csvwrite('b2.csv', b2);

load b3.mat
csvwrite('b3.csv', b3);

load b4.mat
csvwrite('b4.csv', b4);

load b5.mat
csvwrite('b5.csv', b5);

disp('Converting A matrices')

load A1.mat
csvwrite('A1.csv', A1);

load A2.mat
csvwrite('A2.csv', A2);

load A3.mat
csvwrite('A3.csv', A3);

load A4.mat
csvwrite('A4.csv', A4);

load A5.mat
csvwrite('A5.csv', A5);