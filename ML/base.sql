-- 1. Tạo Database mới
CREATE DATABASE AI_Research;
GO

-- Sử dụng database vừa tạo
USE AI_Research;
GO

-- 2. Tạo bảng chuyên mục nghiên cứu
CREATE TABLE Categories (
    CategoryID INT PRIMARY KEY IDENTITY(1,1),
    CategoryName NVARCHAR(100) NOT NULL
);

-- 3. Tạo bảng danh sách dự án
CREATE TABLE Projects (
    ProjectID INT PRIMARY KEY IDENTITY(1,1),
    ProjectName NVARCHAR(255) NOT NULL,
    CategoryID INT,
    StartDate DATE DEFAULT GETDATE(),
    Status NVARCHAR(50) CHECK (Status IN (N'Planning', N'In Progress', N'Completed')),
    Accuracy FLOAT,
    -- Khóa ngoại liên kết với bảng Categories
    FOREIGN KEY (CategoryID) REFERENCES Categories(CategoryID)
);
GO