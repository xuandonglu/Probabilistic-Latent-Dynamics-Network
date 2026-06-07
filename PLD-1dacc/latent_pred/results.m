clear all;clc

load('DPN_d25_fno.mat');
Nx = size(rec, 2);
Nt = size(rec, 3);
dx = 1 / Nx;
dt = 1 / Nt;
T = linspace(0,1.023,Nt)';
X = linspace(0,4,Nx)';
gridT = meshgrid(T, X);
gridX = meshgrid(X, T)';

u1 = squeeze(rec(1, :, :));
u2 = squeeze(rec(2, :, :));
f1 = [gridX(:), gridT(:), u1(:)];
f2 = [gridX(:), gridT(:), u2(:)];
u1 = squeeze(test(1, :, :));
u2 = squeeze(test(2, :, :));
f3 = [gridX(:), gridT(:), u1(:)];
f4 = [gridX(:), gridT(:), u2(:)];
