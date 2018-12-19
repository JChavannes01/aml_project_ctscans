clear all
close all
clc

Warnings = 0;
i_max = 27;
for i = 0:i_max
    disp(['i = ', num2str(i)])
    file_vol = ['..\data\Training_Batch1\volume-',num2str(i),'.nii'];
    file_seg = ['..\data\Training_Batch1\segmentation-',num2str(i),'.nii'];
    CT_vol = niftiread(file_vol);
    CT_seg = niftiread(file_seg);

    info = niftiinfo(file_vol);
    if (info.AdditiveOffset ~= -1024)
        disp('AdditiveOffset not equal to 1024')
        Warnings = Warnings + 1;
    end
    if (info.MultiplicativeScaling ~= 1)
        disp('MultiplicativeScaling not equal to 1')
        Warnings = Warnings + 1;
    end

    Liver = CT_vol(CT_seg == 1);
    Cancer = CT_vol(CT_seg == 2);
    figure
    hold on
    histogram(Liver,800:1:1300, 'FaceColor', 'b', 'EdgeColor', 'none','Normalization','probability');
    histogram(Cancer,800:1:1300, 'FaceColor', 'r', 'EdgeColor', 'none','Normalization','probability');
    legend('Liver', 'Cancer')
    title(['i = ', num2str(i)])
    m_l(i+1) = mean(Liver);
    s_l(i+1) = std(double(Liver));
    m_c(i+1) = mean(Cancer);
    s_c(i+1) = std(double(Cancer));
    disp(['Mean liver = ', num2str(m_l(i+1))])
    disp(['Std liver = ', num2str(s_l(i+1))])
    disp(['Mean cancer = ', num2str(m_c(i+1))])
    disp(['Std cancer = ', num2str(s_c(i+1))])

    %volumeViewer(CT_vol) %To open 3D slicer
end

disp(['Number of warnings: ', num2str(Warnings)])

figure
hold on
plot(0:i_max,m_l)
plot(0:i_max,m_c)
title('Mean value')
legend('Liver', 'Cancer')
figure
hold on
plot(0:i_max,s_l)
plot(0:i_max,s_c)
title('Std value')
legend('Liver', 'Cancer')