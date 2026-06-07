clear all;clc

load('DPN_d10_fno.mat');
rec = rec / 100;
test = test / 100;
Nx = size(rec, 3);
Ny = size(rec, 4);
X = linspace(0,5,Nx)';
Y = linspace(0,5,Ny)';
gridY = meshgrid(Y, X);
gridX = meshgrid(X, Y)';

t = [27, 72, 104, 166, 201];
u1 = squeeze(rec(1, t(1), :, :));
u2 = squeeze(rec(1, t(2), :, :));
u3 = squeeze(rec(1, t(3), :, :));
u4 = squeeze(rec(1, t(4), :, :));
u5 = squeeze(rec(1, t(5), :, :));
f1 = [gridX(:), gridY(:), u1(:), u2(:), u3(:), u4(:), u5(:)];
u1 = squeeze(test(1, t(1), :, :));
u2 = squeeze(test(1, t(2), :, :));
u3 = squeeze(test(1, t(3), :, :));
u4 = squeeze(test(1, t(4), :, :));
u5 = squeeze(test(1, t(5), :, :));
f2 = [gridX(:), gridY(:), u1(:), u2(:), u3(:), u4(:), u5(:)];
