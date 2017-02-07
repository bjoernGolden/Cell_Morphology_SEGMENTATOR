%4.2.16
%Marie Hemmen
%This program reads an image in order to recognize and label the different
%objects (here:yeast cells). The pixel coordinates of the outline are written in a text file
%('LabelAnalysis.txt')


%I=imread('/home/marie/Masterarbeit/Microscopy/20160126_ShmooWT/agar_pad_WT/Image_91.tif');
% I=imread('/home/marie/Masterarbeit/Microscopy/20160219_Mut491_pad/Process_251_MIA.tif');
% I2=imread('/home/marie/Masterarbeit/Microscopy/20160219_Mut491_pad/Process_252_MIA.tif');
% I3=imread('/home/marie/Masterarbeit/Microscopy/20160219_Mut491_pad/Process_253_MIA.tif');

%%write pixel coordinates of the object outlines in fileID
%fileID=fopen('/home/marie/Masterarbeit/Microscopy/20160201_WT/LabelAnalysis.txt','w');

%Txt File in which pixel coordinates of outlines (of segmented cells) with 3 < ratio *Perimeter^2/Area < 400 are written
%FileOutlinesPos=fopen('/home/marie/Master/Outlinecoordinates/Positives/20160219_Mut490_alpha200nM_padwat.txt','w');
%All other pixel coordinates of outlines (of segmented cells)
%FileOutlinesFalsePos=fopen('/home/marie/Master/Outlinecoordinates/FalsePositives/20160219_Mut490_alpha200nM_padwat_outofborderswat.txt','w');

%Txt File in which pixel coordinates of outlines (of segmented cells) with 3 < ratio *Perimeter^2/Area < 400 are written
FileOutlinesPos=fopen('/home/marie/Master/Outlinecoordinates/Positives_newTes/20160804_EKY492_fixiert_0.txt','w');
%All other pixel coordinates of outlines (of segmented cells)
FileOutlinesFalsePos=fopen('/home/marie/Master/Outlinecoordinates/FalsePositives_newTes/20160804_EKY492_fixiert_0.txt','w');


ratio=[];
num=0;

%All files in folder MicroscopeImages
%filelist=dir('/home/marie/Master/MicroscopeImages/*.tif');

filelist=dir('/home/marie/Master/MicroscopeImages/20160804_EKY492_fixiert/0/*.tif');
%hier

%filelist=dir('/home/marie/Master/MicroscopeImages/20160720_EKY360/efi/*.tif')

for i=1:length(filelist)
   filename = strcat('/home/marie/Master/MicroscopeImages/20160804_EKY492_fixiert/0/',filelist(i).name);
   %filename = strcat('/home/marie/Master/MicroscopeImages/20160602_EKY360_alpha10mikroM_fixiert/Process_455.tif')
   %disp('filename: '),disp(filename)
    I = imread(filename);   
    
% fun = @(block_struct) imgaussfilt(I,3);
% I2 = blockproc(I,[100 100],fun);
 %figure;
 %imshow(I);
% figure;
% imshow(I2);
% 
% C = mat2cell(I,[128 128],[128 128]);
%figure, imshow(I),title('original hell');
%show the original image
%figure,imshow(I),title('original image');

%I = imgaussfilt(I,3);
%figure, imshow(I);

%canny filter
[~,threshold]=edge(I,'canny');
 Factor=10;
 BWs=edge(I,'canny',threshold*Factor);

%imdilate connects nearby pixel in order to close an object. 
se90 = strel('line', 3, 90);
se0 = strel('line', 3, 0);
BWsdil = imdilate(BWs, [se90 se0]);
%figure, imshow(BWsdil)

%fill objects
BWsdfill=imfill(BWsdil,'holes');

%remove objects at the border of the image
BWnobord=imclearborder(BWsdfill,4);
%figure, imshow(BWnobord)

%imerode reduces the size of the cells
seD=strel('diamond',1);
BWfinal = imerode(BWnobord,seD);
BWfinal=imerode(BWfinal,seD);
%figure, imshow(BWfinal)

%outline the objects
BWoutline=bwperim(BWfinal,8);%figure,imshow(BWoutline),title('outline');

%overlay original and edited image
Segout=I;
Segout(BWoutline)=255;
%figure,imshow(Segout),title('outlined original image');

%label objects
label2=bwlabel(BWfinal); 
label = bwlabel(BWoutline);
max(max(label));

s = regionprops(label, 'BoundingBox', 'Extrema', 'Centroid');
boxes = cat(1, s.BoundingBox);
left_edge = boxes(:,1);
[sorted, sort_order] = sort(left_edge);
s2 = s(sort_order);
I = im2uint8(Segout);
%I(~Segout) = 200;
%I(Segout) = 240;
set(gcf, 'Position', [100 100 400 300]);
%figure,imshow(I)
hold on
for k = 1:numel(s2)
   centroid = s2(k).Centroid;
   text(centroid(1), centroid(2), sprintf('%d', k));
end
hold off

figure, imshow(label2)
%c is a dictionary
c = containers.Map('KeyType','double','ValueType','any');
label;
Astats = regionprops(label2,'area');
A=[Astats.Area];

%disp('area:'),disp(A)
Pstats=regionprops(label2,'perimeter');
P=[Pstats.Perimeter];

figure, imshow(label2)

for i=1:max(max(label2))
     Plist(i+num)=P(i);
     Alist(i+num)=A(i);
    if P(i)<70 || P(i)>130 ||A(i)<500 || A(i)>1200
       ratio(i+num)=0; 
      
      
    else
       rationew = P(i)^2./A(i);
       ratio(i+num)=rationew; 
      
    end
  

end

for j=1:max(max(label))
   
   if ratio(j+num)>0 && ratio(j+num)<400 %150
        [row,col]=find(label==j);
        matr=[row,col];
       
        len=max(row)-min(row)+2;
        breadth=max(col)-min(col)+2;
        target=uint8(zeros([len breadth]));
        sy=min(col)-1;
        sx=min(row)-1;
        fprintf(FileOutlinesPos,['Label' num2str(j) '\n']);
        for m=1:size(row,1)
            x=row(m,1)-sx;
            y=col(m,1)-sy;
            target(x,y)=I(row(m,1),col(m,1));
            fprintf(FileOutlinesPos,'(%2.f,%3.f),',x,y);

        end
        fprintf(FileOutlinesPos,'\n');
   else
        [row,col]=find(label==j);
        matr=[row,col];
       
        len=max(row)-min(row)+2;
        breadth=max(col)-min(col)+2;
        target=uint8(zeros([len breadth]));
        sy=min(col)-1;
        sx=min(row)-1;
        fprintf(FileOutlinesFalsePos,['Label' num2str(j) '\n']);
        for m=1:size(row,1)
            x=row(m,1)-sx;
            y=col(m,1)-sy;
            target(x,y)=I(row(m,1),col(m,1));
            fprintf(FileOutlinesFalsePos,'(%2.f,%3.f),',x,y);

        end
        fprintf(FileOutlinesFalsePos,'\n');
        end
end
num=num+i;
end

length(Plist);
Plow=30;
Phigh=500;
Plist(Plist<Plow)=[];
Plist(Plist>Phigh)=[];

%bins=round(max(max(label))./4)
%bins = round(2*(max(max(label)))^1./3)
%bins1 = round(2.5*sqrt(length(Plist)))+1
bins1 = round((Phigh-Plow)./6)
figure,subplot(3,1,1)
histogram(Plist,bins1)
xlabel('Perimeter [Pixel]')
ylabel('Segments')
%title('20160219Mut490alpha200nMpad')

alow=200
ahigh=5000
Alist(Alist<alow)=[];
Alist(Alist>ahigh)=[];
bins2 = round((ahigh-alow)./60)
%bins2=round(2.5*sqrt(length(Alist)))+1
subplot(3,1,2)
histogram(Alist,bins2)
xlabel('Area [Pixel^2]')
ylabel('Segments')

rlow=5
rhigh=400
ratio(ratio<rlow)=[];
ratio(ratio>rhigh)=[];
%bins3=round(2.5*sqrt(length(ratio)))+1
bins3 = round((rhigh-rlow)./6)
subplot(3,1,3)
histogram(ratio,bins3)
xlabel('Ratio Perimeter^2/Area [a. u.]')
ylabel('Segments')



Plow=70
Phigh=200
Plist(Plist<Plow)=[];
Plist(Plist>Phigh)=[];

bins=round(max(max(label))./4)
bins = round(2*(max(max(label)))^1./3)
bins1 = round(2.5*sqrt(length(Plist)))+1
bins1 = round((Phigh-Plow)./6)
figure,subplot(3,1,1)
histogram(Plist,bins1)
xlabel('Perimeter')
ylabel('Segments')
%title('20160219Mut490alpha200nMpad')

alow=200
ahigh=5000
Alist(Alist<alow)=[];
Alist(Alist>ahigh)=[];
bins2 = round((ahigh-alow)./6)
bins2=round(2.5*sqrt(length(Alist)))+1
subplot(3,1,2)
histogram(Alist,bins2)
xlabel('Area')
ylabel('Segments')

rlow=0
rhigh=80
ratio(ratio<rlow)=[];
ratio(ratio>rhigh)=[];
bins3=round(2.5*sqrt(length(ratio)))+1
bins3 = round((rhigh-rlow)./6)
subplot(3,1,3)
histogram(ratio,bins3)
xlabel('Ratio Perimeter^2/Area')
ylabel('Segments')
















% for j=1:max(max(label))
%     [row,col]=find(label==j);
%     matr=[row,col];
%     newpic=mat2gray(matr);
%     %figure,imshow(newpic);
% 
%     len=max(row)-min(row)+2;
%     breadth=max(col)-min(col)+2;
%     target=uint8(zeros([len breadth]));
%     sy=min(col)-1;
%     sx=min(row)-1;
%     fprintf(fileID2,['Label' num2str(j) '\n']);
% for i=1:size(row,1)
%     x=row(i,1)-sx;
%     y=col(i,1)-sy;
%     target(x,y)=I(row(i,1),col(i,1));
%     fprintf(fileID2,'(%2.f,%3.f),',x,y);
% 
% end
% fprintf(fileID2,'\n');
% %figure, imshow(target);

% end
% fclose(fileID2);