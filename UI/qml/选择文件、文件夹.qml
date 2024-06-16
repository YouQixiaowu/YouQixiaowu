import QtQuick 6.5
import QtQuick.Controls 6.5
import Qt.labs.platform // FileDialog FolderDialog
Window {
    visible: true
    width: 300
    height: 100
    //
    Button{
        width: 150
        text: "打开文件"
        onClicked: {
            fileDialog.open()
        }
    }
    FileDialog{
        id: fileDialog
        nameFilters: ["json文件 (*.json)"]
        folder: StandardPaths.standardLocations(StandardPaths.PicturesLocation)[0] //默认打开Pictures文件
    }
    // 
    Button{
        x:150
        width: 150
        text: "打开文件夹"
        onClicked: {
            folderDialog.open()
        }
    }
    FolderDialog {
        id: folderDialog
        folder: StandardPaths.standardLocations(StandardPaths.PicturesLocation)[0] //默认打开Pictures文件夹
    }
}
